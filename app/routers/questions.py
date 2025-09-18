from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.includes.proverb_game_core import get_random_proverb, check_proverb_answer

router = APIRouter()

class ProverbAnswerRequest(BaseModel):
    proverb_id: int
    user_answer: str

@router.get("/next")
def get_next_proverb():
    """랜덤 문제 출제"""
    proverb = get_random_proverb()
    if not proverb:
        raise HTTPException(status_code=404, detail="No more proverbs.")
    return proverb

@router.post("/answer")
def submit_answer(req: ProverbAnswerRequest):
    """정답 제출 및 판별"""
    result = check_proverb_answer(req.proverb_id, req.user_answer)
    return result
