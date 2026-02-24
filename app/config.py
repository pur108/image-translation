from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "local"
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    REDIS_URL: str = "redis://localhost:6379/0"

    TRANSLATION_PROVIDER: str = "openai"  # "deepl" or "openai"

    DEEPL_API_KEY: str = ""
    DEEPL_FREE_API: bool = True

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5-nano"

    SUPABASE_URL: str = ""
    SUPABASE_STORAGE_BUCKET: str = "comics"

    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    RESULT_TTL_SECONDS: int = 3600  # 1 hour
    FONT_DIR: str = "./fonts"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
