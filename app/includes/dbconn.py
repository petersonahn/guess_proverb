from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Generator
from app.core.config import settings

SYNC_DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset=utf8mb4"
)

engine = create_engine(
    SYNC_DB_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def healthcheck() -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1
