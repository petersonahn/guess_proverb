"""
🤖 속담 게임 - 난이도 분석 API 라우터

AI 모델을 사용하여 속담의 난이도를 분석하고 점수를 부여하는 API 엔드포인트들을 제공합니다.
"""

import sys
import os
import asyncio
import time
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# 현재 파일의 부모 디렉토리들을 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from app.schemas.proverb import (
    AnalysisRequest, AnalysisResponse, BatchAnalysisRequest, BatchAnalysisResponse,
    ErrorResponse
)

# 전역 변수들은 함수 내에서 import하여 순환 import 방지

router = APIRouter()


def convert_analysis_result_to_response(result: dict, cached: bool = False) -> AnalysisResponse:
    """
    분석 결과를 API 응답 형태로 변환합니다.
    
    Args:
        result: analyzer.analyze_proverb_difficulty()의 결과
        cached: 캐시된 결과인지 여부
        
    Returns:
        AnalysisResponse: API 응답 형태의 분석 결과
    """
    return AnalysisResponse(
        proverb_id=result["proverb_id"],
        full_proverb=result["full_proverb"],
        difficulty_level=result["difficulty_level"],
        confidence=result["confidence"],
        score=result["score"],
        processing_time=result["processing_time"],
        message=result["message"],
        cached=cached
    )


@router.post(
    "/analysis/single",
    response_model=AnalysisResponse,
    summary="개별 속담 난이도 분석",
    description="특정 속담의 난이도를 AI 모델로 분석합니다.",
    responses={
        200: {"description": "분석 성공"},
        404: {"model": ErrorResponse, "description": "속담을 찾을 수 없음"},
        500: {"model": ErrorResponse, "description": "분석 실패"}
    }
)
async def analyze_single_proverb(request: AnalysisRequest):
    """
    🎯 특정 속담의 난이도를 분석합니다.
    
    - **proverb_id**: 분석할 속담의 ID
    - **force_refresh**: true면 캐시를 무시하고 새로 분석
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.difficulty_analyzer:
            raise HTTPException(status_code=500, detail="AI 분석기가 초기화되지 않았습니다")
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        # 속담 존재 여부 확인
        proverb_data = main.proverb_database.get_proverb_by_id(request.proverb_id)
        if not proverb_data:
            raise HTTPException(
                status_code=404,
                detail=f"ID {request.proverb_id}에 해당하는 속담을 찾을 수 없습니다"
            )
        
        # 캐시 확인 (force_refresh가 False인 경우)
        if not request.force_refresh and request.proverb_id in main.analysis_cache:
            main.server_stats["cache_hits"] += 1
            cached_result = main.analysis_cache[request.proverb_id].copy()
            cached_result["processing_time"] = 0.001  # 캐시 조회 시간
            cached_result["message"] += " (캐시됨)"
            return convert_analysis_result_to_response(cached_result, cached=True)
        
        # AI 분석 수행
        analysis_result = main.difficulty_analyzer.analyze_proverb_difficulty(request.proverb_id)
        
        # 분석 실패 처리
        if analysis_result["difficulty_level"] == 0:
            raise HTTPException(
                status_code=500,
                detail=f"분석 실패: {analysis_result['message']}"
            )
        
        # 캐시에 저장
        main.analysis_cache[request.proverb_id] = analysis_result.copy()
        main.server_stats["analysis_count"] += 1
        
        return convert_analysis_result_to_response(analysis_result, cached=False)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@router.post(
    "/analysis/batch",
    response_model=BatchAnalysisResponse,
    summary="배치 속담 난이도 분석",
    description="여러 속담의 난이도를 한번에 분석합니다.",
    responses={
        200: {"description": "배치 분석 성공"},
        500: {"model": ErrorResponse, "description": "배치 분석 실패"}
    }
)
async def analyze_batch_proverbs(request: BatchAnalysisRequest):
    """
    📦 여러 속담의 난이도를 배치로 분석합니다.
    
    - **proverb_ids**: 분석할 속담 ID 목록 (없으면 전체)
    - **force_refresh**: true면 캐시를 무시하고 새로 분석
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        if not main.difficulty_analyzer:
            raise HTTPException(status_code=500, detail="AI 분석기가 초기화되지 않았습니다")
        
        if not main.proverb_database:
            raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다")
        
        start_time = time.time()
        
        # 분석할 속담 ID 목록 결정
        if request.proverb_ids:
            target_ids = request.proverb_ids
        else:
            # 전체 속담 ID 목록 조회
            all_proverbs = main.proverb_database.get_all_proverbs()
            target_ids = [proverb["id"] for proverb in all_proverbs]
        
        results = []
        success_count = 0
        failed_count = 0
        total_score = 0
        
        for proverb_id in target_ids:
            try:
                # 캐시 확인
                if not request.force_refresh and proverb_id in main.analysis_cache:
                    main.server_stats["cache_hits"] += 1
                    cached_result = main.analysis_cache[proverb_id].copy()
                    cached_result["processing_time"] = 0.001
                    cached_result["message"] += " (캐시됨)"
                    
                    response = convert_analysis_result_to_response(cached_result, cached=True)
                    results.append(response)
                    success_count += 1
                    total_score += cached_result["score"]
                    
                else:
                    # 새로 분석
                    analysis_result = main.difficulty_analyzer.analyze_proverb_difficulty(proverb_id)
                    
                    if analysis_result["difficulty_level"] > 0:
                        # 성공
                        main.analysis_cache[proverb_id] = analysis_result.copy()
                        main.server_stats["analysis_count"] += 1
                        
                        response = convert_analysis_result_to_response(analysis_result, cached=False)
                        results.append(response)
                        success_count += 1
                        total_score += analysis_result["score"]
                        
                    else:
                        # 실패
                        failed_count += 1
                        
            except Exception as e:
                failed_count += 1
                print(f"속담 ID {proverb_id} 분석 실패: {str(e)}")
        
        processing_time = time.time() - start_time
        average_score = total_score / success_count if success_count > 0 else 0
        
        return BatchAnalysisResponse(
            total_count=len(target_ids),
            success_count=success_count,
            failed_count=failed_count,
            total_score=total_score,
            average_score=round(average_score, 2),
            processing_time=round(processing_time, 3),
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 분석 실패: {str(e)}")


@router.post(
    "/analysis/preload",
    summary="전체 속담 사전 분석",
    description="모든 속담을 사전에 분석하여 캐시에 저장합니다. (백그라운드 작업)",
    responses={
        202: {"description": "사전 분석 작업 시작됨"},
        500: {"model": ErrorResponse, "description": "사전 분석 시작 실패"}
    }
)
async def preload_all_analysis(background_tasks: BackgroundTasks):
    """
    🚀 모든 속담을 사전에 분석하여 캐시에 저장합니다.
    
    백그라운드 작업으로 실행되므로 즉시 응답을 반환합니다.
    """
    try:
        # 전역 변수 import (순환 import 방지)
        import app.main as main
        
        if not main.difficulty_analyzer:
            raise HTTPException(status_code=500, detail="AI 분석기가 초기화되지 않았습니다")
        
        # 백그라운드 작업으로 전체 분석 실행
        background_tasks.add_task(run_preload_analysis)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "전체 속담 사전 분석 작업이 백그라운드에서 시작되었습니다",
                "status": "started",
                "cache_size": len(main.analysis_cache)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사전 분석 시작 실패: {str(e)}")


async def run_preload_analysis():
    """
    백그라운드에서 실행되는 전체 속담 분석 함수
    """
    try:
        # 전역 변수 import
        import app.main as main
        
        print("🚀 전체 속담 사전 분석 시작...")
        start_time = time.time()
        
        # 배치 분석 실행
        all_results = main.difficulty_analyzer.batch_analyze_all_proverbs()
        
        # 캐시에 저장
        for result in all_results:
            if result["difficulty_level"] > 0:
                main.analysis_cache[result["proverb_id"]] = result
        
        processing_time = time.time() - start_time
        main.server_stats["last_batch_analysis"] = datetime.now().isoformat()
        main.server_stats["analysis_count"] += len([r for r in all_results if r["difficulty_level"] > 0])
        
        print(f"✅ 전체 속담 사전 분석 완료!")
        print(f"   - 분석 완료: {len(main.analysis_cache)}개")
        print(f"   - 처리 시간: {processing_time:.2f}초")
        
    except Exception as e:
        print(f"❌ 사전 분석 실패: {str(e)}")


@router.get(
    "/analysis/cache",
    summary="분석 캐시 상태 조회",
    description="현재 캐시된 분석 결과의 상태를 조회합니다.",
    responses={
        200: {"description": "캐시 상태 조회 성공"}
    }
)
async def get_cache_status():
    """
    💾 현재 분석 캐시의 상태를 조회합니다.
    """
    try:
        # 전역 변수 import
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        # 난이도별 분포 계산
        difficulty_distribution = {}
        total_score = 0
        
        for analysis in main.analysis_cache.values():
            level = analysis.get("difficulty_level", 0)
            if level > 0:
                difficulty_distribution[level] = difficulty_distribution.get(level, 0) + 1
                total_score += analysis.get("score", 0)
        
        # 전체 속담 수 조회
        total_proverbs = main.proverb_database.get_proverb_count() if main.proverb_database else 0
        
        return {
            "cache_size": len(main.analysis_cache),
            "total_proverbs": total_proverbs,
            "coverage": f"{(len(main.analysis_cache) / max(total_proverbs, 1)) * 100:.1f}%",
            "difficulty_distribution": difficulty_distribution,
            "total_score": total_score,
            "average_score": round(total_score / max(len(main.analysis_cache), 1), 2),
            "last_batch_analysis": main.server_stats.get("last_batch_analysis"),
            "stats": {
                "total_requests": main.server_stats["total_requests"],
                "cache_hits": main.server_stats["cache_hits"],
                "analysis_count": main.server_stats["analysis_count"],
                "hit_rate": f"{(main.server_stats['cache_hits'] / max(main.server_stats['total_requests'], 1)) * 100:.1f}%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 상태 조회 실패: {str(e)}")


@router.delete(
    "/analysis/cache",
    summary="분석 캐시 초기화",
    description="모든 캐시된 분석 결과를 삭제합니다.",
    responses={
        200: {"description": "캐시 초기화 성공"}
    }
)
async def clear_cache():
    """
    🗑️ 모든 캐시된 분석 결과를 삭제합니다.
    """
    try:
        # 전역 변수 import
        import app.main as main
        
        main.server_stats["total_requests"] += 1
        
        old_size = len(main.analysis_cache)
        main.analysis_cache.clear()
        
        return {
            "message": "분석 캐시가 초기화되었습니다",
            "cleared_count": old_size,
            "current_size": len(main.analysis_cache)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 초기화 실패: {str(e)}")
