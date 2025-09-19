"""
📋 속담 게임 - Pydantic 모델 정의

API 요청/응답을 위한 데이터 모델들을 정의합니다.
FastAPI의 자동 문서화와 타입 검증을 위해 사용됩니다.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProverbBase(BaseModel):
    """속담 기본 모델"""
    id: int = Field(..., description="속담 고유 ID")
    question: str = Field(..., description="속담 앞부분 (문제)")
    answer: str = Field(..., description="속담 뒷부분 (정답)")
    hint: str = Field(..., description="힌트")


class ProverbResponse(BaseModel):
    """속담 조회 응답 모델"""
    id: int = Field(..., description="속담 고유 ID")
    question: str = Field(..., description="속담 앞부분 (문제)")
    answer: str = Field(..., description="속담 뒷부분 (정답)")
    hint: str = Field(..., description="힌트")
    full_proverb: str = Field(..., description="완전한 속담 텍스트")
    difficulty_level: Optional[int] = Field(None, description="난이도 (1=쉬움, 2=보통, 3=어려움)")
    score: Optional[int] = Field(None, description="점수 (1-3점)")
    confidence: Optional[float] = Field(None, description="AI 분석 신뢰도 (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "question": "가는 말이 고와야",
                "answer": "오는 말이 곱다",
                "hint": "말의 왕래와 관련된 속담입니다.",
                "full_proverb": "가는 말이 고와야 오는 말이 곱다",
                "difficulty_level": 1,
                "score": 1,
                "confidence": 0.85
            }
        }


class ProverbListResponse(BaseModel):
    """속담 목록 조회 응답 모델"""
    total_count: int = Field(..., description="전체 속담 개수")
    page: int = Field(..., description="현재 페이지")
    limit: int = Field(..., description="페이지당 항목 수")
    proverbs: List[ProverbResponse] = Field(..., description="속담 목록")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 84,
                "page": 1,
                "limit": 10,
                "proverbs": [
                    {
                        "id": 1,
                        "question": "가는 말이 고와야",
                        "answer": "오는 말이 곱다",
                        "hint": "말의 왕래와 관련된 속담입니다.",
                        "full_proverb": "가는 말이 고와야 오는 말이 곱다",
                        "difficulty_level": 1,
                        "score": 1,
                        "confidence": 0.85
                    }
                ]
            }
        }


class AnalysisRequest(BaseModel):
    """난이도 분석 요청 모델"""
    proverb_id: int = Field(..., description="분석할 속담 ID", gt=0)
    force_refresh: bool = Field(False, description="캐시를 무시하고 새로 분석할지 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "proverb_id": 1,
                "force_refresh": False
            }
        }


class AnalysisResponse(BaseModel):
    """난이도 분석 응답 모델"""
    proverb_id: int = Field(..., description="속담 ID")
    full_proverb: str = Field(..., description="완전한 속담 텍스트")
    difficulty_level: int = Field(..., description="난이도 (1=쉬움, 2=보통, 3=어려움)")
    confidence: float = Field(..., description="AI 분석 신뢰도 (0-1)")
    score: int = Field(..., description="점수 (1-3점)")
    processing_time: float = Field(..., description="처리 시간 (초)")
    message: str = Field(..., description="분석 결과 메시지")
    cached: bool = Field(..., description="캐시된 결과인지 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "proverb_id": 1,
                "full_proverb": "가는 말이 고와야 오는 말이 곱다",
                "difficulty_level": 1,
                "confidence": 0.85,
                "score": 1,
                "processing_time": 0.15,
                "message": "분석 완료 - 쉬움 (1점)",
                "cached": False
            }
        }


class BatchAnalysisRequest(BaseModel):
    """배치 분석 요청 모델"""
    proverb_ids: Optional[List[int]] = Field(None, description="분석할 속담 ID 목록 (없으면 전체)")
    force_refresh: bool = Field(False, description="캐시를 무시하고 새로 분석할지 여부")
    
    class Config:
        json_schema_extra = {
            "example": {
                "proverb_ids": [1, 2, 3, 4, 5],
                "force_refresh": False
            }
        }


class BatchAnalysisResponse(BaseModel):
    """배치 분석 응답 모델"""
    total_count: int = Field(..., description="총 분석 대상 개수")
    success_count: int = Field(..., description="성공한 분석 개수")
    failed_count: int = Field(..., description="실패한 분석 개수")
    total_score: int = Field(..., description="총 점수")
    average_score: float = Field(..., description="평균 점수")
    processing_time: float = Field(..., description="총 처리 시간 (초)")
    results: List[AnalysisResponse] = Field(..., description="개별 분석 결과")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 5,
                "success_count": 5,
                "failed_count": 0,
                "total_score": 11,
                "average_score": 2.2,
                "processing_time": 0.75,
                "results": []
            }
        }


class ServerStats(BaseModel):
    """서버 통계 모델"""
    startup_time: Optional[str] = Field(None, description="서버 시작 시간")
    total_requests: int = Field(..., description="총 요청 수")
    cache_hits: int = Field(..., description="캐시 히트 수")
    analysis_count: int = Field(..., description="분석 수행 횟수")
    last_batch_analysis: Optional[str] = Field(None, description="마지막 배치 분석 시간")
    cache_size: int = Field(..., description="캐시 크기")
    hit_rate: str = Field(..., description="캐시 히트율")


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: str = Field(..., description="에러 타입")
    message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 정보")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "NotFound",
                "message": "속담을 찾을 수 없습니다",
                "detail": "ID 999에 해당하는 속담이 데이터베이스에 존재하지 않습니다."
            }
        }
