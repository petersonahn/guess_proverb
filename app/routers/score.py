import os
# ✅ transformers가 TensorFlow/Flax를 건드리지 않도록 차단
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select
from unicodedata import normalize as ucnorm
import numpy as np

from app.includes.dbconn import get_db
from app.models.proverb import Proverb

# ✅ Trainer를 우회하기 위해 __init__ 대신 직접 모듈 경로에서 임포트
#    (이렇게 하면 sentence_transformers.trainer -> transformers.Trainer 경로를 타지 않습니다)
try:
    from sentence_transformers.SentenceTransformer import SentenceTransformer  # 핵심
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# (옵션) 폴백용 – 모델 로드가 실패하면 RapidFuzz로 최소 기능 유지
try:
    from rapidfuzz import fuzz
    _RF_AVAILABLE = True
except Exception:
    _RF_AVAILABLE = False

router = APIRouter(prefix="/score", tags=["score"])

# ---- 전역 모델 로드 ----
_model = None
if _ST_AVAILABLE:
    try:
        _model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    except Exception:
        _model = None

class ScoreRequest(BaseModel):
    id: int = Field(..., ge=1)
    answer: str = Field(..., min_length=1)

class ScoreResponse(BaseModel):
    similarity: float  # 0~100
    awarded: int       # similarity를 반올림한 점수
    correct: bool
    threshold: int = 60

def _clean_text(s: str) -> str:
    # 임베딩 특성상 과도한 삭제는 지양하고 NFC정규화 + 트리밍만
    return ucnorm("NFC", (s or "").strip())

def _semantic_similarity(a: str, b: str) -> float:
    """
    문장 임베딩 코사인 유사도 -> 0~100 스케일
    """
    if _model is None:
        raise RuntimeError("Embedding model is not available")
    embs = _model.encode([a, b], normalize_embeddings=True)
    cos = float(np.dot(embs[0], embs[1]))  # -1 ~ 1
    return (cos + 1.0) / 2.0 * 100.0      # 0 ~ 100

def _fuzzy_similarity(a: str, b: str) -> float:
    # 폴백: 문자열 유사도(0~100)
    if not _RF_AVAILABLE:
        return 0.0
    return float(fuzz.ratio(a, b))

@router.post("", response_model=ScoreResponse)
def score_answer(payload: ScoreRequest, db: Session = Depends(get_db)):
    row = db.execute(select(Proverb).where(Proverb.id == payload.id)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")

    gold = _clean_text(row.answer)       # 정답(뒷부분)
    user = _clean_text(payload.answer)   # 사용자 입력(뒷부분)

    threshold = 60

    if _model is not None:
        sim = _semantic_similarity(gold, user)   # 0~100
    else:
        # 임베딩이 불가한 환경(설치/가속기 문제 등)에 대한 안전장치
        sim = _fuzzy_similarity(gold, user)

    awarded = int(round(sim))
    correct = sim >= threshold

    return ScoreResponse(similarity=float(sim), awarded=awarded, correct=bool(correct), threshold=threshold)
