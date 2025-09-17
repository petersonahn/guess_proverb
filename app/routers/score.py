# app/routers/score.py
import os
# ✅ Transformers가 TensorFlow/Flax 로드하지 않도록 차단
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

# ✅ Trainer 경로 우회 import (불필요한 무거운 의존성 로드 방지)
try:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# (옵션) 임베딩 모델 로드 실패 시 폴백
try:
    from rapidfuzz import fuzz
    _RF_AVAILABLE = True
except Exception:
    _RF_AVAILABLE = False

router = APIRouter(prefix="/score", tags=["score"])

# ---- 전역 모델 및 정답 임베딩 캐시 ----
_model = None
_gold_cache: dict[int, np.ndarray] = {}  # id -> gold embedding (unit vector)

if _ST_AVAILABLE:
    try:
        # ✅ 한국어 짧은 구절/문장 임베딩에 적합한 KoSimCSE
        _model = SentenceTransformer("BM-K/KoSimCSE-roberta")
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
    # 의미 보존: NFC 정규화 + trim만
    return ucnorm("NFC", (s or "").strip())

def _get_gold_vec(db: Session, qid: int) -> tuple[str, np.ndarray]:
    """
    문제 id의 정답 텍스트와 임베딩(unit vector)을 반환.
    캐시에 없으면 DB에서 읽어 1회 임베딩 후 캐시에 저장.
    """
    if qid in _gold_cache:
        # 정답 텍스트는 필요 없지만, 로깅/디버그용으로 함께 반환 가능
        row = db.execute(select(Proverb.answer).where(Proverb.id == qid)).first()
        gold_text = _clean_text(row[0]) if row else ""
        return gold_text, _gold_cache[qid]

    row = db.execute(select(Proverb).where(Proverb.id == qid)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")
    gold_text = _clean_text(row.answer)

    if _model is None:
        raise RuntimeError("Embedding model is not available")

    # normalize_embeddings=True -> 이미 L2 정규화된 unit vector 반환
    vec = _model.encode([gold_text], normalize_embeddings=True)[0]
    _gold_cache[qid] = vec
    return gold_text, vec

def _semantic_similarity(gold_vec: np.ndarray, user_text: str) -> float:
    """
    unit vector(gold) · unit vector(user) = cos ∈ [-1,1] → [0,100]
    """
    if _model is None:
        raise RuntimeError("Embedding model is not available")
    user_vec = _model.encode([_clean_text(user_text)], normalize_embeddings=True)[0]
    cos = float(np.dot(gold_vec, user_vec))
    return (cos + 1.0) / 2.0 * 100.0

def _fuzzy_similarity(a: str, b: str) -> float:
    # 폴백: 문자열 유사도(0~100)
    if not _RF_AVAILABLE:
        return 0.0
    return float(fuzz.ratio(a, b))

@router.post("", response_model=ScoreResponse)
def score_answer(payload: ScoreRequest, db: Session = Depends(get_db)):
    threshold = 60

    if _model is not None:
        gold_text, gold_vec = _get_gold_vec(db, payload.id)
        sim = _semantic_similarity(gold_vec, payload.answer)
    else:
        # 임베딩이 불가한 환경(설치/가속기 이슈 등) 대비 안전장치
        row = db.execute(select(Proverb.answer).where(Proverb.id == payload.id)).first()
        if not row:
            raise HTTPException(status_code=404, detail="Question not found")
        gold_text = _clean_text(row[0])
        sim = _fuzzy_similarity(gold_text, payload.answer)

    awarded = int(round(sim))
    correct = sim >= threshold
    return ScoreResponse(similarity=float(sim), awarded=awarded, correct=bool(correct), threshold=threshold)
