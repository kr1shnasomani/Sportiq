from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_path: Path = Field(default=BASE_DIR / "model" / "TrackNet.pth", alias="MODEL_PATH")
    device: str = Field(default="auto", alias="DEVICE")  # auto|cpu|mps|cuda
    fast_mode: bool = Field(default=False, alias="FAST_MODE")
    allowed_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
    ], alias="ALLOWED_ORIGINS")

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
