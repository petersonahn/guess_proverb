# app/routers/score.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select
from unicodedata import normalize as ucnorm
import unicodedata
import numpy as np
import re
from typing import Dict, Tuple

from app.includes.dbconn import get_db
from app.models.proverb import Proverb

# SentenceTransformer를 직접 경로에서 임포트(불필요한 Trainer 경로 회피)
try:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

# RapidFuzz (문자 유사도)
try:
    from rapidfuzz import fuzz
    _RF_AVAILABLE = True
except Exception:
    _RF_AVAILABLE = False

router = APIRouter(prefix="/score", tags=["score"])

# ===== KoSimCSE 모델 로드 =====
_model = None
if _ST_AVAILABLE:
    try:
        _model = SentenceTransformer("BM-K/KoSimCSE-roberta")
    except Exception:
        _model = None

# ===== 캐시: 정답 임베딩/정규화 =====
_gold_vec_cache: Dict[int, np.ndarray] = {}           # id -> unit vector
_gold_norm_cache: Dict[int, str] = {}                 # id -> 정답 표면형 정규화
_gold_jamo_cache: Dict[int, str] = {}                 # id -> 정답 자모열
_gold_bigrams_cache: Dict[int, set] = {}              # id -> 정답 bigram set

# ===== 유틸 =====
PUNCT_RE = re.compile(r"[.,!?~\-_/\(\)\[\]\"'“”‘’:;·…]+", re.UNICODE)
WS_RE = re.compile(r"\s+", re.UNICODE)

def _clean_text_meaning(s: str) -> str:
    """의미 평가는 과도한 제거 없이 NFC + trim만"""
    return ucnorm("NFC", (s or "").strip())

def _clean_text_surface(s: str) -> str:
    """문자 유사도용: 공백/문장부호 제거"""
    s = _clean_text_meaning(s)
    s = PUNCT_RE.sub("", s)
    s = WS_RE.sub("", s)
    return s

# 한글 자모 분해 (초성/중성/종성)
# 참고: 유니코드 한글 음절 범위
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def _decompose_hangul(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            si = code - 0xAC00
            l = si // 588
            v = (si % 588) // 28
            t = si % 28
            out.append(CHO[l]); out.append(JUNG[v])
            if t != 0:
                out.append(JONG[t])
        else:
            # 한글 자모(낱자)나 기타 문자는 그대로 유지
            out.append(ch)
    return "".join(out)

def _bigrams(s: str) -> set:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s)-1)}

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ===== 임베딩/표면형 신호 계산 =====
def _get_gold(db: Session, qid: int) -> Tuple[str, str, str, set, np.ndarray]:
    """
    정답 텍스트(의미/표면형), 자모열, bigrams, 임베딩(unit) 반환.
    필요한 값은 캐시에 저장.
    """
    # 표면형/자모/bigram 캐시
    if qid not in _gold_norm_cache or qid not in _gold_jamo_cache or qid not in _gold_bigrams_cache:
        row = db.execute(select(Proverb.answer).where(Proverb.id == qid)).first()
        if not row:
            raise HTTPException(status_code=404, detail="Question not found")
        gold = _clean_text_meaning(row[0])
        gold_norm = _clean_text_surface(gold)
        gold_jamo = _decompose_hangul(gold_norm)
        gold_bi = _bigrams(gold_norm)
        _gold_norm_cache[qid] = gold_norm
        _gold_jamo_cache[qid] = gold_jamo
        _gold_bigrams_cache[qid] = gold_bi
        # 의미용 원문 반환 위해 gold 자체도 보유
        _gold_norm_cache[f"{qid}__gold_raw"] = gold

    gold_raw = _gold_norm_cache.get(f"{qid}__gold_raw", "")
    gold_norm = _gold_norm_cache[qid]
    gold_jamo = _gold_jamo_cache[qid]
    gold_bi = _gold_bigrams_cache[qid]

    # 임베딩 캐시
    gold_vec = None
    if _model is not None:
        if qid not in _gold_vec_cache:
            vec = _model.encode([gold_raw], normalize_embeddings=True)[0]
            _gold_vec_cache[qid] = vec
        gold_vec = _gold_vec_cache[qid]

    return gold_raw, gold_norm, gold_jamo, gold_bi, gold_vec

def _semantic_sim(gold_vec: np.ndarray, user_text: str) -> float:
    """KoSimCSE 임베딩 코사인 유사도 → 0~100"""
    user = _clean_text_meaning(user_text)
    uvec = _model.encode([user], normalize_embeddings=True)[0]
    cos = float(np.dot(gold_vec, uvec))
    return (cos + 1.0) / 2.0 * 100.0

def _char_sim(gold_norm: str, user_norm: str) -> float:
    """문자 유사도 (RapidFuzz ratio) 0~100"""
    if not _RF_AVAILABLE:
        return 0.0
    return float(fuzz.ratio(gold_norm, user_norm))

def _jamo_sim(gold_jamo: str, user_jamo: str) -> float:
    """자모 분해 후 문자 유사도 0~100 (오타/받침 차이 보정)"""
    if not _RF_AVAILABLE:
        return 0.0
    return float(fuzz.ratio(gold_jamo, user_jamo))

def _ngram_sim(gold_bi: set, user_bi: set) -> float:
    """문자 bigram Jaccard 0~100"""
    return _jaccard(gold_bi, user_bi) * 100.0

def _length_penalty(gold_norm: str, user_norm: str) -> float:
    """길이 큰 차이 패널티 (너무 짧거나 길면 약간 감점) → 0.9~1.0"""
    lg, lu = max(len(gold_norm), 1), max(len(user_norm), 1)
    ratio = min(lg, lu) / max(lg, lu)  # 0~1
    # 0.9 ~ 1.0 사이 완만한 패널티
    return 0.9 + 0.1 * ratio

# ===== 요청/응답 모델 =====
class ScoreRequest(BaseModel):
    id: int = Field(..., ge=1)
    answer: str = Field(..., min_length=1)

class ScoreResponse(BaseModel):
    similarity: float   # 최종 0~100
    awarded: int        # 부여 점수
    correct: bool
    threshold: int = 70

@router.post("", response_model=ScoreResponse)
def score_answer(payload: ScoreRequest, db: Session = Depends(get_db)):
    # 정답 리소스
    gold_raw, gold_norm, gold_jamo, gold_bi, gold_vec = _get_gold(db, payload.id)

    # 사용자 전처리
    user_raw  = _clean_text_meaning(payload.answer)
    user_norm = _clean_text_surface(user_raw)
    user_jamo = _decompose_hangul(user_norm)
    user_bi   = _bigrams(user_norm)

    # 개별 신호
    sem = _semantic_sim(gold_vec, user_raw) if (gold_vec is not None and _model is not None) else None
    ch  = _char_sim(gold_norm, user_norm)
    jm  = _jamo_sim(gold_jamo, user_jamo)
    ng  = _ngram_sim(gold_bi, user_bi)

    # 가중치 (의미 위주 + 문자 보강)
    w_sem, w_char, w_ng = 0.58, 0.22, 0.20
    # 모델 폴백 시 가중치 재정의(문자 기반만)
    if sem is None:
        w_sem, w_char, w_ng = 0.0, 0.55, 0.45

    # 자모 유사도는 문자 유사도의 보정으로 사용(더 높은 쪽 채택)
    char_combo = max(ch, jm)

    # 길이 페널티
    lp = _length_penalty(gold_norm, user_norm)

    # 최종 점수
    base = (w_sem * (sem or 0.0)) + (w_char * char_combo) + (w_ng * ng)
    final = max(0.0, min(100.0, base * lp))

    threshold = 70
    awarded = int(round(final))
    correct = final >= threshold

    return ScoreResponse(similarity=float(final), awarded=awarded, correct=bool(correct), threshold=threshold)
