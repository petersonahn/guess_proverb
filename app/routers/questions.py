from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func
import random

from app.includes.dbconn import get_db
from app.models.proverb import Proverb
from app.schemas.proverb import ProverbOut

router = APIRouter(prefix="/questions", tags=["questions"])

@router.get("", response_model=List[ProverbOut])
def list_questions(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    # 무작위 순서로 일부 추출 (간단 무작위)
    total = db.execute(select(func.count()).select_from(Proverb)).scalar_one()
    q = db.execute(select(Proverb)).scalars().all()
    items = q
    random.shuffle(items)
    items = items[:limit]
    return items
