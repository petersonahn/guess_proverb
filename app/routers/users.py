from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.includes.dbconn import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserOut

router = APIRouter(prefix="/users", tags=["users"])

@router.get("", response_model=List[UserOut])
def get_users(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    # total_score 내림차순, 같으면 created_at 오름차순
    stmt = (
        select(User)
        .order_by(User.total_score.desc(), User.created_at.asc())
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()

@router.post("", response_model=UserOut)
def upsert_user(payload: UserCreate, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")
    if len(username) > 20:
        raise HTTPException(status_code=400, detail="username must be <= 20 chars")

    existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if existing:
        # 기존 정책 유지: 새 점수가 더 크면 갱신 (누적이 필요하면 += 로 변경)
        if payload.total_score > (existing.total_score or 0):
            existing.total_score = payload.total_score
            db.add(existing)
            db.commit()
            db.refresh(existing)
        return existing

    row = User(username=username, total_score=payload.total_score)
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(row)
    return row
