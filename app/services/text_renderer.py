import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.services.ocr import TextRegion

from pythainlp.tokenize import word_tokenize as thai_word_tokenize


class TextRenderer:
    def __init__(self, settings: Settings):
        self.font_dir = settings.FONT_DIR
        self._fonts: dict[str, str] = {}
        self._load_font_paths()

    def _load_font_paths(self):
        thai_font = os.path.join(self.font_dir, "NotoSansThai-Regular.ttf")
        en_font = os.path.join(self.font_dir, "NotoSans-Regular.ttf")

        if os.path.exists(thai_font):
            self._fonts["th"] = thai_font
        if os.path.exists(en_font):
            self._fonts["en"] = en_font

    def _get_font(self, lang: str, size: int) -> ImageFont.FreeTypeFont:
        font_path = self._fonts.get(lang) or self._fonts.get("en")
        if not font_path:
            raise RuntimeError(
                f"No font file found for lang '{lang}' in {self.font_dir}"
            )
        return ImageFont.truetype(font_path, size)

    def _get_bbox_rect(self, bbox: list[list[int]]) -> tuple[int, int, int, int]:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _estimate_bg_color(
        self, image: np.ndarray, bbox: list[list[int]]
    ) -> tuple[int, int, int]:
        x1, y1, x2, y2 = self._get_bbox_rect(bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return (255, 255, 255)
        mean = cv2.mean(region)[:3]
        return (int(mean[2]), int(mean[1]), int(mean[0]))  # BGR -> RGB

    def _get_text_color(self, bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
        luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)

    def _wrap_text(
        self, text: str, lang: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> list[str]:
        if lang == "th" and thai_word_tokenize:
            tokens = thai_word_tokenize(text)
        else:
            tokens = text.split(" ")

        sep = "" if lang == "th" else " "
        lines: list[str] = []
        current_line = ""

        for token in tokens:
            test_line = current_line + sep + token if current_line else token
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]

            if text_width <= max_width or not current_line:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = token

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    def _fit_font_size(
        self,
        text: str,
        lang: str,
        max_width: int,
        max_height: int,
        min_size: int = 8,
        max_size: int = 72,
    ) -> int:
        best_size = min_size

        for size in range(max_size, min_size - 1, -1):
            font = self._get_font(lang, size)
            lines = self._wrap_text(text, lang, font, max_width)

            line_height = size * (1.4 if lang == "th" else 1.2)
            total_height = line_height * len(lines)

            if total_height <= max_height:
                max_line_width = 0
                for line in lines:
                    bb = font.getbbox(line)
                    max_line_width = max(max_line_width, bb[2] - bb[0])

                if max_line_width <= max_width:
                    best_size = size
                    break

        return best_size

    def render(
        self,
        image: np.ndarray,
        regions: list[TextRegion],
        translated_texts: list[str],
        target_lang: str,
    ) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_image)

        for region, translated in zip(regions, translated_texts):
            x1, y1, x2, y2 = self._get_bbox_rect(region.bbox)
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            padding = 2
            available_width = box_width - padding * 2
            available_height = box_height - padding * 2

            if available_width <= 0 or available_height <= 0:
                continue

            font_size = self._fit_font_size(
                translated, target_lang, available_width, available_height
            )
            font = self._get_font(target_lang, font_size)

            bg_color = self._estimate_bg_color(image, region.bbox)
            text_color = self._get_text_color(bg_color)

            lines = self._wrap_text(translated, target_lang, font, available_width)
            line_height = font_size * (1.4 if target_lang == "th" else 1.2)
            total_text_height = line_height * len(lines)

            start_y = y1 + (box_height - total_text_height) / 2

            for i, line in enumerate(lines):
                bb = font.getbbox(line)
                line_width = bb[2] - bb[0]
                line_x = x1 + (box_width - line_width) / 2
                line_y = start_y + i * line_height

                draw.text((line_x, line_y), line, font=font, fill=text_color)

        result = np.array(pil_image)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
