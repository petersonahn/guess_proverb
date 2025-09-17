from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "proverb_game"
    DB_USER: str = "root"
    DB_PASS: str = "1234"

    API_PREFIX: str = "/api"

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
