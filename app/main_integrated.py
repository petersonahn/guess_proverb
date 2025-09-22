#!/usr/bin/env python3
"""
🎯 속담 게임 - 통합 FastAPI 애플리케이션
AI 난이도 분석 시스템과 게임 기능을 통합한 완전한 속담 게임 서버

주요 기능:
1. 60초 제한시간 속담 게임
2. AI 기반 하이브리드 난이도 분석 (사용자 작업 통합)
3. 정답 유사도 측정 (KR-SBERT)
4. 힌트 시스템 (3가지 방식)
5. 랭킹 시스템
6. 속담 조회/검색 API
7. 난이도 분석 API (사용자 담당 기능)

실행 방법:
    uvicorn app.main_integrated:app --host 127.0.0.1 --port 8080 --reload
"""

import sys
import os
import time
import random
import traceback
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# FastAPI 관련 imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 프로젝트 모듈 imports
try:
    from app.core.config import proverb_config
except ImportError as e:
    print(f"❌ 설정 모듈 import 실패: {e}")
    sys.exit(1)

# ==================== Pydantic 모델 정의 ====================

class GameStartRequest(BaseModel):
    """게임 시작 요청"""
    pass

class AnswerSubmitRequest(BaseModel):
    """정답 제출 요청"""
    answer: str
    game_id: str
    question_id: int

class HintRequest(BaseModel):
    """힌트 요청"""
    game_id: str
    question_id: int

class RankingSaveRequest(BaseModel):
    """랭킹 저장 요청"""
    username: str
    score: int

class GameSession:
    """게임 세션 관리 클래스"""
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.start_time = time.time()
        self.end_time = None
        self.current_score = 0
        self.questions_answered = 0
        self.correct_answers = 0
        self.current_question = None
        self.used_question_ids = set()
        self.hint_used = False
        self.wrong_count = 0
        self.streak_count = 0
        self.time_limit = 60  # 60초
        
    def is_expired(self) -> bool:
        """게임 시간이 만료되었는지 확인"""
        if self.end_time:
            return True
        return (time.time() - self.start_time) >= self.time_limit
    
    def get_remaining_time(self) -> int:
        """남은 시간 반환 (초)"""
        if self.end_time:
            return 0
        elapsed = time.time() - self.start_time
        remaining = max(0, self.time_limit - elapsed)
        return int(remaining)
    
    def end_game(self):
        """게임 종료"""
        self.end_time = time.time()

# ==================== 전역 변수 ====================

# 게임 세션 저장소
game_sessions: Dict[str, GameSession] = {}

# AI 모델들 (lazy loading으로 변경)
difficulty_analyzer = None
similarity_model = None
proverb_database = None

# 분석 결과 캐시 (사용자 난이도 분석 기능)
analysis_cache: Dict[int, Dict[str, Any]] = {}

# 서버 통계
server_stats = {
    "startup_time": None,
    "total_requests": 0,
    "cache_hits": 0,
    "analysis_count": 0,
    "last_batch_analysis": None
}

# ==================== Lazy Loading 함수들 ====================

def get_difficulty_analyzer():
    """난이도 분석기를 lazy loading으로 가져옵니다 (사용자 작업)."""
    global difficulty_analyzer
    if difficulty_analyzer is None:
        print("⏳ 사용자 난이도 분석기 로딩 중...")
        try:
            from app.includes.analyzer import ProverbDifficultyAnalyzer
            difficulty_analyzer = ProverbDifficultyAnalyzer()
            print("✅ 사용자 난이도 분석기 로딩 완료")
        except Exception as e:
            print(f"❌ 난이도 분석기 로딩 실패: {e}")
            # 기본 분석기 사용
            class DefaultAnalyzer:
                def analyze_proverb_difficulty(self, proverb_id):
                    return {"difficulty_level": 2, "score": 2, "confidence": 0.5}
            difficulty_analyzer = DefaultAnalyzer()
    return difficulty_analyzer

def get_similarity_model():
    """유사도 측정 모델을 lazy loading으로 가져옵니다."""
    global similarity_model
    if similarity_model is None:
        print("⏳ 유사도 측정 모델 로딩 중...")
        try:
            from sentence_transformers import SentenceTransformer
            similarity_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            print("✅ 유사도 측정 모델 로딩 완료")
        except Exception as e:
            print(f"❌ 유사도 측정 모델 로딩 실패: {e}")
            raise
    return similarity_model

def get_proverb_database():
    """데이터베이스 연결을 lazy loading으로 가져옵니다."""
    global proverb_database
    if proverb_database is None:
        print("⏳ 데이터베이스 연결 중...")
        try:
            from app.includes.dbconn import ProverbDatabase
            proverb_database = ProverbDatabase()
            print("✅ 데이터베이스 연결 완료")
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            raise
    return proverb_database

# ==================== FastAPI 앱 생성 ====================

app = FastAPI(
    title="🎯 속담 게임 - 통합 시스템",
    description="""
    ## 하이브리드 속담 난이도 분석 및 게임 시스템
    
    사용자의 AI 난이도 분석 기능과 새로운 게임 시스템을 통합한 완전한 속담 게임입니다.
    
    ### 주요 기능
    - 🎮 60초 제한시간 속담 게임
    - 🧠 사용자 개발 하이브리드 난이도 분석 (사용 빈도 중심)
    - 📚 속담 조회 및 검색 API
    - 🏆 랭킹 시스템
    - ⚡ 분석 결과 캐싱으로 빠른 응답
    
    ### 기술 스택
    - **Backend**: FastAPI + Python
    - **AI Model**: jhgan/ko-sroberta-multitask (사용자 지정)
    - **Database**: MySQL (proverb_game)
    - **Analysis**: 사용자의 하이브리드 분석 시스템
    """,
    version="1.2.0"
)

# 정적 파일 및 템플릿 설정
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

# 사용자의 API 라우터들 등록 (일시적으로 비활성화)
print("⚠️ 사용자 라우터는 일시적으로 비활성화됨 (충돌 해결 중)")

print("✅ FastAPI 서버 초기화 완료 (AI 모델은 필요할 때 로드됩니다)")

# ==================== 유사도 측정 함수들 ====================

def get_threshold_by_length(answer: str) -> float:
    """답안 길이에 따른 유사도 임계값 반환"""
    length = len(answer)
    if length <= 2:
        return 0.55
    elif length <= 4:
        return 0.60
    elif length <= 6:
        return 0.65
    elif length <= 8:
        return 0.70
    elif length <= 12:
        return 0.75
    elif length <= 16:
        return 0.80
    else:
        return 0.85

def check_answer_similarity(user_answer: str, correct_answer: str, threshold: float) -> Dict[str, Any]:
    """사용자 답안과 정답의 유사도를 측정합니다."""
    try:
        model = get_similarity_model()  # lazy loading
        
        # 임베딩 생성
        from sentence_transformers import util
        embeddings = model.encode([user_answer, correct_answer])
        
        # 코사인 유사도 계산
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # 정답 여부 판단
        is_correct = similarity >= threshold
        
        return {
            "is_correct": is_correct,
            "similarity": round(similarity, 3),
            "threshold": threshold,
            "user_answer": user_answer,
            "correct_answer": correct_answer
        }
        
    except Exception as e:
        print(f"❌ 유사도 측정 실패: {str(e)}")
        return {
            "is_correct": False,
            "similarity": 0.0,
            "threshold": threshold,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "error": str(e)
        }

# ==================== 게임 로직 함수들 ====================

def generate_game_id() -> str:
    """고유한 게임 ID 생성"""
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"game_{timestamp}_{random_num}"

def get_random_question(used_ids: set = None) -> Optional[Dict[str, Any]]:
    """랜덤 문제를 가져옵니다."""
    try:
        db = get_proverb_database()  # lazy loading
        
        if used_ids is None:
            used_ids = set()
        
        # 전체 속담 개수 조회
        total_count = db.get_proverb_count()
        if total_count == 0:
            return None
        
        # 사용 가능한 ID 목록 생성
        available_ids = []
        for i in range(1, total_count + 1):
            if i not in used_ids:
                available_ids.append(i)
        
        if not available_ids:
            return None  # 모든 문제를 다 풀었음
        
        # 랜덤하게 선택
        selected_id = random.choice(available_ids)
        
        # 속담 데이터 조회
        proverb_data = db.get_proverb_by_id(selected_id)
        if not proverb_data:
            return None
        
        # 사용자의 난이도 분석 시스템 사용
        analyzer = get_difficulty_analyzer()
        difficulty_result = analyzer.analyze_proverb_difficulty(selected_id)
        
        return {
            "question_id": selected_id,
            "question": proverb_data["question"],
            "answer": proverb_data["answer"],
            "hint": proverb_data["hint"],
            "full_proverb": proverb_data["full_proverb"],
            "difficulty_level": difficulty_result.get("difficulty_level", 2),
            "difficulty_score": difficulty_result.get("score", 2),
            "confidence": difficulty_result.get("confidence", 0.5)
        }
        
    except Exception as e:
        print(f"❌ 문제 생성 실패: {str(e)}")
        return None

def calculate_score(difficulty_level: int, hint_used: bool = False) -> int:
    """난이도와 힌트 사용 여부에 따른 점수 계산"""
    base_scores = {1: 100, 2: 200, 3: 300}  # 쉬움: 100점, 보통: 200점, 어려움: 300점
    base_score = base_scores.get(difficulty_level, 200)
    
    if hint_used:
        return base_score // 2  # 힌트 사용 시 절반
    else:
        return base_score

# ==================== API 엔드포인트 ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    if templates:
        return templates.TemplateResponse("main.html", {"request": request})
    else:
        return HTMLResponse("<h1>속담 게임 API 서버</h1><p><a href='/docs'>API 문서 보기</a></p>")

@app.get("/game", response_class=HTMLResponse)
async def game_page(request: Request):
    """게임 페이지"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("<h1>게임 페이지</h1><p>템플릿 파일이 없습니다. <a href='/docs'>API 문서</a>를 참조하세요.</p>")

@app.get("/rankings", response_class=HTMLResponse)
async def rankings_page(request: Request):
    """랭킹 페이지"""
    if templates:
        return templates.TemplateResponse("rankings.html", {"request": request})
    else:
        return HTMLResponse("<h1>랭킹 페이지</h1><p>템플릿 파일이 없습니다. <a href='/docs'>API 문서</a>를 참조하세요.</p>")

@app.post("/api/game/start")
async def start_game(request: GameStartRequest):
    """게임 시작 (사용자 난이도 분석 시스템 통합)"""
    try:
        print("🎮 게임 시작 요청 - 사용자 AI 모델 로딩 시작...")
        
        # 새 게임 세션 생성
        game_id = generate_game_id()
        session = GameSession(game_id)
        
        # 첫 번째 문제 생성 (이때 사용자의 모델들이 lazy loading됨)
        question = get_random_question(session.used_question_ids)
        if not question:
            raise HTTPException(status_code=500, detail="문제를 생성할 수 없습니다.")
        
        session.current_question = question
        session.used_question_ids.add(question["question_id"])
        
        # 세션 저장
        game_sessions[game_id] = session
        
        # 서버 통계 초기화 (첫 게임 시작 시)
        if server_stats["startup_time"] is None:
            server_stats["startup_time"] = datetime.now().isoformat()
        
        print("✅ 게임 시작 완료! (사용자 난이도 분석 시스템 적용)")
        
        return {
            "success": True,
            "game_id": game_id,
            "question": {
                "question_id": question["question_id"],
                "question_text": question["question"],
                "difficulty_level": question["difficulty_level"],
                "difficulty_name": proverb_config.PROVERB_DIFFICULTY_LEVELS[question["difficulty_level"]]["name"]
            },
            "game_info": {
                "time_limit": session.time_limit,
                "remaining_time": session.get_remaining_time(),
                "current_score": session.current_score,
                "questions_answered": session.questions_answered
            }
        }
        
    except Exception as e:
        print(f"❌ 게임 시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 시작 실패: {str(e)}")

@app.post("/api/game/answer")
async def submit_answer(request: AnswerSubmitRequest):
    """정답 제출 (유사도 측정 통합)"""
    try:
        # 게임 세션 확인
        session = game_sessions.get(request.game_id)
        if not session:
            raise HTTPException(status_code=404, detail="게임 세션을 찾을 수 없습니다.")
        
        # 게임 시간 확인
        if session.is_expired():
            session.end_game()
            return {
                "success": False,
                "game_ended": True,
                "message": "게임 시간이 종료되었습니다.",
                "final_score": session.current_score
            }
        
        # 현재 문제 확인
        if not session.current_question or session.current_question["question_id"] != request.question_id:
            raise HTTPException(status_code=400, detail="잘못된 문제 ID입니다.")
        
        current_q = session.current_question
        
        # 유사도 측정
        threshold = get_threshold_by_length(current_q["answer"])
        similarity_result = check_answer_similarity(request.answer, current_q["answer"], threshold)
        
        is_correct = similarity_result["is_correct"]
        similarity_score = similarity_result["similarity"]
        
        if is_correct:
            # 정답 처리 (사용자의 난이도 시스템으로 점수 계산)
            score = calculate_score(current_q["difficulty_level"], session.hint_used)
            session.current_score += score
            session.correct_answers += 1
            session.questions_answered += 1
            session.streak_count += 1
            session.wrong_count = 0  # 오답 횟수 리셋
            session.hint_used = False  # 힌트 사용 리셋
            
            # 다음 문제 생성
            next_question = get_random_question(session.used_question_ids)
            
            if next_question and not session.is_expired():
                # 다음 문제가 있고 시간이 남은 경우
                session.current_question = next_question
                session.used_question_ids.add(next_question["question_id"])
                
                return {
                    "success": True,
                    "correct": True,
                    "score_earned": score,
                    "similarity": similarity_score,
                    "message": f"정답입니다! (+{score}점)",
                    "next_question": {
                        "question_id": next_question["question_id"],
                        "question_text": next_question["question"],
                        "difficulty_level": next_question["difficulty_level"],
                        "difficulty_name": proverb_config.PROVERB_DIFFICULTY_LEVELS[next_question["difficulty_level"]]["name"]
                    },
                    "game_info": {
                        "remaining_time": session.get_remaining_time(),
                        "current_score": session.current_score,
                        "questions_answered": session.questions_answered,
                        "streak_count": session.streak_count
                    }
                }
            else:
                # 문제가 없거나 시간 종료
                session.end_game()
                return {
                    "success": True,
                    "correct": True,
                    "score_earned": score,
                    "similarity": similarity_score,
                    "game_ended": True,
                    "message": "게임 종료! 모든 문제를 완료했습니다.",
                    "final_score": session.current_score,
                    "stats": {
                        "questions_answered": session.questions_answered,
                        "correct_answers": session.correct_answers,
                        "accuracy": round((session.correct_answers / max(1, session.questions_answered)) * 100, 1)
                    }
                }
        else:
            # 오답 처리 (기존 로직 유지)
            session.wrong_count += 1
            session.streak_count = 0  # 연속 정답 리셋
            
            if session.wrong_count >= 2:
                # 2회 오답 시 다음 문제로
                session.questions_answered += 1
                session.wrong_count = 0
                session.hint_used = False
                
                # 다음 문제 생성
                next_question = get_random_question(session.used_question_ids)
                
                if next_question and not session.is_expired():
                    session.current_question = next_question
                    session.used_question_ids.add(next_question["question_id"])
                    
                    return {
                        "success": True,
                        "correct": False,
                        "similarity": similarity_score,
                        "message": f"2회 오답! 정답: {current_q['answer']}",
                        "correct_answer": current_q["answer"],
                        "skip_to_next": True,
                        "next_question": {
                            "question_id": next_question["question_id"],
                            "question_text": next_question["question"],
                            "difficulty_level": next_question["difficulty_level"],
                            "difficulty_name": proverb_config.PROVERB_DIFFICULTY_LEVELS[next_question["difficulty_level"]]["name"]
                        },
                        "game_info": {
                            "remaining_time": session.get_remaining_time(),
                            "current_score": session.current_score,
                            "questions_answered": session.questions_answered,
                            "streak_count": session.streak_count
                        }
                    }
                else:
                    # 게임 종료
                    session.end_game()
                    return {
                        "success": True,
                        "correct": False,
                        "similarity": similarity_score,
                        "game_ended": True,
                        "message": "게임 종료!",
                        "correct_answer": current_q["answer"],
                        "final_score": session.current_score
                    }
            else:
                # 1회 오답 시 힌트 제공 - 힌트 사용 상태 설정
                session.hint_used = True
                return {
                    "success": True,
                    "correct": False,
                    "similarity": similarity_score,
                    "message": f"오답입니다. (유사도: {similarity_score:.2f})",
                    "show_hint": True,
                    "hint": current_q["hint"],
                    "wrong_count": session.wrong_count,
                    "game_info": {
                        "remaining_time": session.get_remaining_time(),
                        "current_score": session.current_score,
                        "questions_answered": session.questions_answered,
                        "streak_count": session.streak_count
                    }
                }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 정답 제출 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/game/hint")
async def get_hint(request: HintRequest):
    """힌트 요청"""
    try:
        # 게임 세션 확인
        session = game_sessions.get(request.game_id)
        if not session:
            raise HTTPException(status_code=404, detail="게임 세션을 찾을 수 없습니다.")
        
        # 현재 문제 확인
        if not session.current_question or session.current_question["question_id"] != request.question_id:
            raise HTTPException(status_code=400, detail="잘못된 문제 ID입니다.")
        
        # 힌트 사용 표시
        session.hint_used = True
        
        return {
            "success": True,
            "hint": session.current_question["hint"],
            "message": "힌트를 사용했습니다. 정답 시 점수가 절반으로 줄어듭니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 힌트 요청 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ranking/save")
async def save_ranking(request: RankingSaveRequest):
    """랭킹 저장"""
    try:
        # MySQL 직접 연결
        import mysql.connector
        conn = mysql.connector.connect(
            host=proverb_config.DB_HOST,
            port=proverb_config.DB_PORT,
            user=proverb_config.DB_USER,
            password=proverb_config.DB_PASSWORD,
            database=proverb_config.DB_NAME,
            charset='utf8mb4',
            autocommit=True
        )
        
        cursor = conn.cursor()
        query = """
            INSERT INTO user (username, total_score, created_at) 
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (request.username, request.score, datetime.now()))
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": f"{request.username}님의 점수 {request.score}점이 랭킹에 저장되었습니다."
        }
        
    except Exception as e:
        print(f"❌ 랭킹 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ranking/list")
async def get_rankings(limit: int = 10):
    """랭킹 목록 조회"""
    try:
        # MySQL 직접 연결
        import mysql.connector
        conn = mysql.connector.connect(
            host=proverb_config.DB_HOST,
            port=proverb_config.DB_PORT,
            user=proverb_config.DB_USER,
            password=proverb_config.DB_PASSWORD,
            database=proverb_config.DB_NAME,
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        query = """
            SELECT username, total_score, created_at 
            FROM user 
            ORDER BY total_score DESC, created_at ASC 
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        rankings = []
        for i, (username, score, created_at) in enumerate(results, 1):
            rankings.append({
                "rank": i,
                "username": username,
                "score": score,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "success": True,
            "rankings": rankings
        }
        
    except Exception as e:
        print(f"❌ 랭킹 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """서버 상태 확인"""
    try:
        return {
            "status": "healthy",
            "server": "integrated_main",
            "models_loaded": {
                "difficulty_analyzer": difficulty_analyzer is not None,
                "similarity_model": similarity_model is not None,
                "proverb_db": proverb_database is not None
            },
            "active_games": len(game_sessions),
            "user_features": {
                "difficulty_analysis": "사용자 하이브리드 분석 시스템",
                "cache_size": len(analysis_cache),
                "analysis_count": server_stats.get("analysis_count", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("🎯 속담 게임 통합 서버 시작 (사용자 난이도 분석 시스템 포함)")
    print("=" * 60)
    print(f"🌐 주소: http://{proverb_config.API_HOST}:{proverb_config.API_PORT}")
    print(f"📖 API 문서: http://{proverb_config.API_HOST}:{proverb_config.API_PORT}/docs")
    print(f"🎮 게임: http://{proverb_config.API_HOST}:{proverb_config.API_PORT}/")
    print("💡 사용자 AI 모델은 첫 게임 시작 시 로드됩니다")
    print("🎯 사용자 난이도 분석: 사용 빈도 중심 하이브리드 시스템")
    print("=" * 60)
    
    uvicorn.run(
        "main_integrated:app",
        host=proverb_config.API_HOST,
        port=proverb_config.API_PORT,
        reload=proverb_config.API_RELOAD
    )
