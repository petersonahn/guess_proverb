"""
📚 속담 게임 - 속담 조회 API 라우터

데이터베이스에 저장된 속담 데이터를 조회하는 API 엔드포인트들을 제공합니다.
"""

import sys
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

# 현재 파일의 부모 디렉토리들을 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from app.schemas.proverb import (
    ProverbResponse, ProverbListResponse, ErrorResponse
)

# 전역 변수들은 함수 내에서 import하여 순환 import 방지

router = APIRouter()


def get_proverb_with_analysis(proverb_data: dict) -> ProverbResponse:
    """
    속담 데이터에 분석 결과를 추가하여 반환합니다.
    
    Args:
        proverb_data: 데이터베이스에서 조회한 속담 데이터
        
    Returns:
        ProverbResponse: 분석 결과가 포함된 속담 응답
    """
    # 전역 변수 import (순환 import 방지)
    import app.main as main
    
    # 기본 속담 정보
    response_data = {
        "id": proverb_data["id"],
        "question": proverb_data["question"],
        "answer": proverb_data["answer"],
        "hint": proverb_data["hint"],
        "full_proverb": proverb_data["full_proverb"]
    }
    
    # 캐시된 분석 결과가 있으면 추가
    if proverb_data["id"] in main.analysis_cache:
        cached_analysis = main.analysis_cache[proverb_data["id"]]
        response_data.update({
            "difficulty_level": cached_analysis.get("difficulty_level"),
            "score": cached_analysis.get("score"),
            "confidence": cached_analysis.get("confidence")
        })
    
    return ProverbResponse(**response_data)


@router.get(
    "/proverbs",
    response_model=ProverbListResponse,
    summary="속담 목록 조회",
    description="데이터베이스에 저장된 속담 목록을 페이지네이션으로 조회합니다.",
    responses={
        200: {"description": "속담 목록 조회 성공"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"}
    }
)
async def get_proverbs(
    page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
    limit: int = Query(10, ge=1, le=100, description="페이지당 항목 수 (1-100)"),
    difficulty: Optional[int] = Query(None, ge=1, le=3, description="난이도 필터 (1=쉬움, 2=보통, 3=어려움)")
):
    """
    📚 속담 목록을 조회합니다.
    
    - **page**: 페이지 번호 (1부터 시작)
    - **limit**: 페이지당 항목 수 (1-100)
    - **difficulty**: 난이도 필터 (선택사항)
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        # 전체 개수 조회
        total_count = main.proverb_database.get_proverb_count()
        
        # 페이지네이션 계산
        offset = (page - 1) * limit
        
        # 속담 목록 조회
        proverbs_data = main.proverb_database.get_proverbs_batch(offset, limit)
        
        # 난이도 필터 적용 (캐시된 분석 결과 기준)
        if difficulty is not None:
            filtered_proverbs = []
            for proverb in proverbs_data:
                if proverb["id"] in main.analysis_cache:
                    cached_analysis = main.analysis_cache[proverb["id"]]
                    if cached_analysis.get("difficulty_level") == difficulty:
                        filtered_proverbs.append(proverb)
            proverbs_data = filtered_proverbs
        
        # 응답 데이터 구성
        proverbs_response = [get_proverb_with_analysis(proverb) for proverb in proverbs_data]
        
        return ProverbListResponse(
            total_count=total_count,
            page=page,
            limit=limit,
            proverbs=proverbs_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"속담 목록 조회 실패: {str(e)}")


@router.get(
    "/proverbs/{proverb_id}",
    response_model=ProverbResponse,
    summary="개별 속담 조회",
    description="특정 ID의 속담을 조회합니다.",
    responses={
        200: {"description": "속담 조회 성공"},
        404: {"model": ErrorResponse, "description": "속담을 찾을 수 없음"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"}
    }
)
async def get_proverb(proverb_id: int):
    """
    🔍 특정 ID의 속담을 조회합니다.
    
    - **proverb_id**: 조회할 속담의 ID
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        # 속담 조회
        proverb_data = main.proverb_database.get_proverb_by_id(proverb_id)
        
        if not proverb_data:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {proverb_id}에 해당하는 속담을 찾을 수 없습니다"
            )
        
        # 분석 결과와 함께 반환
        return get_proverb_with_analysis(proverb_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"속담 조회 실패: {str(e)}")


@router.get(
    "/proverbs/random",
    response_model=ProverbResponse,
    summary="랜덤 속담 조회",
    description="무작위로 선택된 속담을 조회합니다.",
    responses={
        200: {"description": "랜덤 속담 조회 성공"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"}
    }
)
async def get_random_proverb(
    difficulty: Optional[int] = Query(None, ge=1, le=3, description="원하는 난이도 (1=쉬움, 2=보통, 3=어려움)")
):
    """
    🎲 무작위로 선택된 속담을 조회합니다.
    
    - **difficulty**: 원하는 난이도 (선택사항, 캐시된 분석 결과 기준)
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        # 난이도 필터가 있으면 해당 난이도의 속담들만 필터링
        if difficulty is not None and main.analysis_cache:
            # 캐시에서 해당 난이도의 속담 ID들 찾기
            filtered_ids = [
                proverb_id for proverb_id, analysis in main.analysis_cache.items()
                if analysis.get("difficulty_level") == difficulty
            ]
            
            if not filtered_ids:
                raise HTTPException(
                    status_code=404, 
                    detail=f"난이도 {difficulty}에 해당하는 속담을 찾을 수 없습니다"
                )
            
            # 랜덤으로 ID 선택
            import random
            random_id = random.choice(filtered_ids)
            proverb_data = main.proverb_database.get_proverb_by_id(random_id)
        else:
            # 전체에서 랜덤 선택
            all_proverbs = main.proverb_database.get_all_proverbs()
            if not all_proverbs:
                raise HTTPException(status_code=404, detail="속담 데이터가 없습니다")
            
            import random
            proverb_data = random.choice(all_proverbs)
        
        return get_proverb_with_analysis(proverb_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"랜덤 속담 조회 실패: {str(e)}")


@router.get(
    "/proverbs/search",
    response_model=ProverbListResponse,
    summary="속담 검색",
    description="키워드로 속담을 검색합니다.",
    responses={
        200: {"description": "속담 검색 성공"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"}
    }
)
async def search_proverbs(
    q: str = Query(..., min_length=1, description="검색 키워드"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(10, ge=1, le=100, description="페이지당 항목 수")
):
    """
    🔍 키워드로 속담을 검색합니다.
    
    - **q**: 검색 키워드 (속담 텍스트에서 검색)
    - **page**: 페이지 번호
    - **limit**: 페이지당 항목 수
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        # 전체 속담에서 키워드 검색
        all_proverbs = main.proverb_database.get_all_proverbs()
        search_results = []
        
        for proverb in all_proverbs:
            # 속담 전체 텍스트에서 키워드 검색 (대소문자 무시)
            full_text = f"{proverb['question']} {proverb['answer']} {proverb['hint']}"
            if q.lower() in full_text.lower():
                search_results.append(proverb)
        
        # 페이지네이션 적용
        total_count = len(search_results)
        offset = (page - 1) * limit
        paginated_results = search_results[offset:offset + limit]
        
        # 응답 데이터 구성
        proverbs_response = [get_proverb_with_analysis(proverb) for proverb in paginated_results]
        
        return ProverbListResponse(
            total_count=total_count,
            page=page,
            limit=limit,
            proverbs=proverbs_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"속담 검색 실패: {str(e)}")
