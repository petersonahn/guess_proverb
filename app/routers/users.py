# ... 위 생략 ...
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.includes.dbconn import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserOut

router = APIRouter(prefix="/users", tags=["users"])

@router.get("", response_model=List[UserOut])
def get_users(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    stmt = (
        select(User)
        .order_by(User.total_score.desc(), User.created_at.asc())
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()

@router.get("/top", response_model=List[UserOut])
def get_top_users(size: int = Query(5, ge=1, le=50), db: Session = Depends(get_db)):
    """
    상위 N명 랭킹. 점수 내림차순, 동일 점수는 등록일(created_at) 빠른 순.
    """
    stmt = (
        select(User)
        .order_by(User.total_score.desc(), User.created_at.asc())
        .limit(size)
    )
    return db.execute(stmt).scalars().all()

@router.post("", response_model=UserOut)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")
    if len(username) > 20:
        raise HTTPException(status_code=400, detail="username must be <= 20 chars")

    exists = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자명입니다.")

    row = User(username=username, total_score=payload.total_score)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="이미 존재하는 사용자명입니다.")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="db error")

    db.refresh(row)
    return row
