#!/usr/bin/env python3
"""
🎯 속담 게임 FastAPI 서버
AI 속담 난이도 분석 및 점수 캐싱 시스템

주요 기능:
- 데이터베이스 속담 데이터 조회
- AI 모델로 속담 난이도 자동 분석 
- 3단계 난이도별 점수 부여 (1점=쉬움, 2점=보통, 3점=어려움)
- 메모리 캐시로 빠른 응답 제공
- 배치 처리로 전체 속담 사전 분석

실행 방법:
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
    
API 문서:
    http://localhost:8001/docs
"""

import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# 현재 파일의 부모 디렉토리들을 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 기존 모듈들 import
try:
    from app.core.config import proverb_config
    from app.includes.analyzer import ProverbDifficultyAnalyzer
    from app.includes.dbconn import ProverbDatabase
    from app.includes.utils import validate_database_connection, get_system_status
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("해결 방법: 프로젝트 루트에서 실행하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("proverb_api")

# 전역 변수들
difficulty_analyzer: Optional[ProverbDifficultyAnalyzer] = None
proverb_database: Optional[ProverbDatabase] = None
analysis_cache: Dict[int, Dict[str, Any]] = {}
server_stats = {
    "startup_time": None,
    "total_requests": 0,
    "cache_hits": 0,
    "analysis_count": 0,
    "last_batch_analysis": None
}


async def initialize_services():
    """
    🚀 서버 시작 시 필요한 서비스들을 초기화합니다.
    """
    global difficulty_analyzer, proverb_database
    
    logger.info("🎯 속담 게임 API 서버 초기화 중...")
    
    try:
        # 1. 데이터베이스 연결 확인
        logger.info("🗄️ 데이터베이스 연결 확인 중...")
        db_status = validate_database_connection()
        if not db_status["success"]:
            raise Exception(f"데이터베이스 연결 실패: {db_status['message']}")
        
        logger.info(f"✅ 데이터베이스 연결 성공 - {db_status['proverb_count']}개 속담")
        
        # 2. 데이터베이스 인스턴스 생성
        proverb_database = ProverbDatabase()
        
        # 3. AI 분석기 초기화 (시간이 걸림)
        logger.info("🤖 AI 난이도 분석기 초기화 중... (시간이 걸릴 수 있습니다)")
        difficulty_analyzer = ProverbDifficultyAnalyzer()
        
        logger.info("✅ AI 분석기 초기화 완료")
        
        # 4. 서버 시작 시간 기록
        from datetime import datetime
        server_stats["startup_time"] = datetime.now().isoformat()
        
        logger.info("🎉 서버 초기화 완료!")
        
    except Exception as e:
        logger.error(f"❌ 서버 초기화 실패: {str(e)}")
        raise


async def shutdown_services():
    """
    🛑 서버 종료 시 리소스를 정리합니다.
    """
    global difficulty_analyzer, proverb_database
    
    logger.info("🛑 서버 종료 중...")
    
    try:
        if difficulty_analyzer:
            difficulty_analyzer.close()
            logger.info("✅ AI 분석기 종료 완료")
        
        if proverb_database:
            proverb_database.close()
            logger.info("✅ 데이터베이스 연결 종료 완료")
            
        logger.info("✅ 서버 종료 완료")
        
    except Exception as e:
        logger.error(f"⚠️ 서버 종료 중 오류: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션 생명주기 관리
    """
    # 시작 시 초기화
    await initialize_services()
    
    yield
    
    # 종료 시 정리
    await shutdown_services()


# FastAPI 앱 생성
app = FastAPI(
    title="🎯 속담 게임 API",
    description="""
    ## 속담 난이도 분석 및 점수 캐싱 시스템
    
    AI 모델(jhgan/ko-sroberta-multitask)을 사용하여 한국 속담의 난이도를 자동으로 분석하고,
    3단계 난이도별 점수를 부여하는 백엔드 API 서버입니다.
    
    ### 주요 기능
    - 📚 데이터베이스 속담 데이터 조회
    - 🤖 AI 모델로 속담 난이도 자동 분석
    - 🏆 3단계 난이도별 점수 부여 (1점=쉬움, 2점=보통, 3점=어려움)
    - ⚡ 메모리 캐시로 빠른 응답 제공
    - 📦 배치 처리로 전체 속담 사전 분석
    
    ### 기술 스택
    - **Backend**: FastAPI + Python
    - **AI Model**: jhgan/ko-sroberta-multitask
    - **Database**: MySQL (proverb_game.proverb)
    - **Cache**: 인메모리 딕셔너리
    """,
    version="1.0.0",
    contact={
        "name": "속담 게임 개발팀",
        "email": "dev@proverb-game.com"
    },
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경용, 운영에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 미들웨어: 요청 통계 수집
@app.middleware("http")
async def collect_request_stats(request, call_next):
    """요청 통계를 수집하는 미들웨어"""
    server_stats["total_requests"] += 1
    response = await call_next(request)
    return response


# API 라우터들 등록 (사용자 담당 기능만)
from app.routers import proverbs, analysis

app.include_router(
    proverbs.router, 
    prefix="/api/v1", 
    tags=["📚 속담 조회"],
    responses={404: {"description": "Not found"}}
)

app.include_router(
    analysis.router, 
    prefix="/api/v1", 
    tags=["🤖 난이도 분석"],
    responses={404: {"description": "Not found"}}
)


@app.get("/", summary="서버 상태 확인")
async def root():
    """
    🏠 서버 상태를 확인하는 루트 엔드포인트
    """
    return {
        "message": "🎯 속담 게임 API 서버",
        "status": "running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "stats": server_stats
    }


@app.get("/health", summary="헬스 체크")
async def health_check():
    """
    🏥 서버와 연결된 서비스들의 상태를 확인합니다.
    """
    try:
        # 데이터베이스 상태 확인
        db_status = validate_database_connection()
        
        # 시스템 상태 확인
        system_status = get_system_status()
        
        # AI 분석기 상태 확인
        analyzer_status = {
            "initialized": difficulty_analyzer is not None,
            "model_loaded": difficulty_analyzer.sentence_model is not None if difficulty_analyzer else False
        }
        
        return {
            "status": "healthy",
            "timestamp": server_stats["startup_time"],
            "services": {
                "database": db_status,
                "analyzer": analyzer_status,
                "system": system_status
            },
            "cache": {
                "size": len(analysis_cache),
                "hit_rate": f"{(server_stats['cache_hits'] / max(server_stats['total_requests'], 1)) * 100:.1f}%"
            },
            "stats": server_stats
        }
        
    except Exception as e:
        logger.error(f"헬스 체크 실패: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "stats": server_stats
            }
        )


if __name__ == "__main__":
    print("🎯 속담 게임 - AI 난이도 분석 API 서버")
    print("=" * 50)
    print("📋 사용자 담당 기능:")
    print("  - 속담 데이터 조회 API")
    print("  - AI 난이도 분석 API") 
    print("  - 분석 결과 캐싱")
    print()
    print("⚠️  이 서버는 Python 직접 실행용입니다.")
    print("💡 FastAPI 앱 객체가 생성되었습니다: 'app'")
    print("📚 API 문서는 웹 서버 실행 시 /docs에서 확인 가능")
    print()
    print("🔧 개발/테스트 방법:")
    print("  1. 구성 테스트: python quick_server_test.py")
    print("  2. 통합 테스트: python test_system.py") 
    print("  3. 점수 확인: python check_scores.py")
    print()
    
    # 기본 구성 확인
    try:
        print("✅ FastAPI 앱 초기화 완료")
        print(f"✅ 등록된 라우터: {len([r for r in app.routes if hasattr(r, 'path')])}개 엔드포인트")
        print("✅ 속담 난이도 분석 API 서버 준비 완료!")
    except Exception as e:
        print(f"❌ 초기화 오류: {str(e)}")
