import re

import deepl
from openai import OpenAI

from app.config import Settings
from app.errors import QuotaExceededError, TranslationError

LANG_MAP_SOURCE = {
    "th": "TH",
    "en": "EN",
}

LANG_MAP_TARGET = {
    "th": "TH",
    "en": "EN-US",
}

LANG_NAMES = {
    "th": "Thai",
    "en": "English",
}


class DeepLTranslationService:
    def __init__(self, settings: Settings):
        server_url = None
        if settings.DEEPL_FREE_API:
            server_url = "https://api-free.deepl.com"
        self.translator = deepl.Translator(
            settings.DEEPL_API_KEY, server_url=server_url
        )

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        if not texts:
            return []

        deepl_source = LANG_MAP_SOURCE.get(source_lang)
        deepl_target = LANG_MAP_TARGET.get(target_lang)

        if not deepl_source or not deepl_target:
            raise TranslationError(
                "UNSUPPORTED_LANGUAGE",
                f"Language pair {source_lang}->{target_lang} is not supported",
                400,
            )

        try:
            results = self.translator.translate_text(
                texts,
                source_lang=deepl_source,
                target_lang=deepl_target,
            )
        except deepl.QuotaExceededException:
            raise QuotaExceededError()
        except deepl.DeepLException as e:
            raise TranslationError("TRANSLATION_FAILED", f"DeepL API error: {e}", 502)

        if isinstance(results, list):
            return [r.text for r in results]
        return [results.text]


class OpenAITranslationService:
    def __init__(self, settings: Settings):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        if not texts:
            return []

        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        system_prompt = (
            f"You are a translator. Translate the following texts from "
            f"{source_name} to {target_name}. Return ONLY the translations, "
            f"one per line, preserving the original order. "
            f"Do not add numbering or explanations."
        )
        user_content = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )

            content = response.choices[0].message.content
            if content is None:
                raise TranslationError(
                    "TRANSLATION_FAILED", "OpenAI returned empty response", 502
                )
            raw = content.strip()
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            lines = [re.sub(r"^\[\d+\]\s*", "", line) for line in lines]

            if len(lines) != len(texts):
                raise TranslationError(
                    "TRANSLATION_MISMATCH",
                    f"Expected {len(texts)} translations, got {len(lines)}",
                    502,
                )
            return lines

        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError("TRANSLATION_FAILED", f"OpenAI API error: {e}", 502)


def create_translation_service(
    settings: Settings,
) -> DeepLTranslationService | OpenAITranslationService:
    provider = settings.TRANSLATION_PROVIDER.lower()
    if provider == "deepl":
        return DeepLTranslationService(settings)
    if provider == "openai":
        return OpenAITranslationService(settings)
    raise ValueError(f"Unknown translation provider: {provider}")


# Backward-compatible alias so TranslationService(settings) still works
TranslationService = create_translation_service
