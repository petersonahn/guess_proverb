import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time
import re
from typing import List, Tuple, Dict
from functools import lru_cache
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 향상된 유사도 검사 시스템 임포트
try:
    from .enhanced_similarity_checker import EnhancedProverbSimilarityChecker
    ENHANCED_CHECKER_AVAILABLE = True
except ImportError:
    ENHANCED_CHECKER_AVAILABLE = False
    print("경고: 향상된 유사도 검사 모듈을 찾을 수 없습니다. 기본 로직을 사용합니다.")

# AI 학습 기반 지능적 시스템 임포트
try:
    from .ai_learning_system import IntelligentProverbChecker, ProverbDataLearner
    AI_LEARNING_AVAILABLE = True
except ImportError:
    AI_LEARNING_AVAILABLE = False
    print("경고: AI 학습 시스템을 찾을 수 없습니다. 기본 로직을 사용합니다.")

# 통합된 임계값 설정 (직접 구현)
UNIFIED_THRESHOLD_AVAILABLE = True

# 통합된 임계값 매핑 테이블 (공통 사용) - 동의어 허용 고려
THRESHOLD_MAP = [
    (2, 0.96), (3, 0.95), (4, 0.93), (5, 0.91), # 짧은 속담들 - 동의어 허용
    (6, 0.89), (8, 0.87), (10, 0.85),            # 중간 길이 - 동의어 허용
    (15, 0.80), (20, 0.75)                       # 긴 속담들 - 동의어 허용
]

# 동적 핵심 단어 검사 모듈은 아래에 직접 구현됨
DYNAMIC_KEYWORD_CHECK_AVAILABLE = True

# ===== 상수 정의 (AI 학습 시스템으로 대체됨) =====
# 기존 하드코딩된 상수들은 AI 학습 시스템이 자동으로 학습하여 처리합니다.
# ProverbDataLearner 클래스가 proverb 테이블의 실제 데이터를 분석하여
# 동물 분류, 의미적 반대 관계, 핵심 단어 등을 자동으로 학습합니다.

# 백업용 상수들은 AI 학습 시스템으로 대체됨


# ===== 동적 키워드 검사 시스템 =====
class ProverbKeywordChecker:
    """속담별 핵심 단어 동적 검사 클래스"""
    
    def __init__(self, db_config: Dict[str, str] = None):
        """
        초기화
        
        Args:
            db_config: 데이터베이스 연결 설정
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'root', 
            'password': 'root1234',
            'database': 'proverb_game'
        }
    
    def get_db_connection(self):
        """데이터베이스 연결 생성"""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            return None
    
    @lru_cache(maxsize=128)
    def get_critical_words_by_answer(self, answer: str) -> List[Tuple[str, float]]:
        """
        답안으로 해당 속담의 핵심 단어와 패널티 계수를 조회
        
        Args:
            answer: 속담 답안 (예: "쳇바퀴 돌듯")
            
        Returns:
            List[Tuple[str, float]]: [(핵심단어, 패널티계수), ...]
        """
        conn = self.get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            
            # proverb_keywords 테이블이 있는지 확인
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = 'proverb_keywords'
            """, (self.db_config['database'],))
            
            if cursor.fetchone()[0] == 0:
                logger.warning("proverb_keywords 테이블이 존재하지 않습니다. 기본 로직을 사용합니다.")
                return []
            
            # 핵심 단어 조회
            query = """
                SELECT pk.keyword, pk.penalty_factor, pk.importance_level
                FROM proverb p
                JOIN proverb_keywords pk ON p.id = pk.proverb_id
                WHERE p.answer = %s
                ORDER BY 
                    CASE pk.importance_level 
                        WHEN 'critical' THEN 1 
                        WHEN 'high' THEN 2 
                        WHEN 'medium' THEN 3 
                        WHEN 'low' THEN 4 
                    END
            """
            
            cursor.execute(query, (answer,))
            results = cursor.fetchall()
            
            return [(keyword, penalty_factor) for keyword, penalty_factor, _ in results]
            
        except mysql.connector.Error as e:
            logger.error(f"핵심 단어 조회 실패: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def detect_dynamic_critical_word_missing(self, answer: str, user_input: str) -> float:
        """
        데이터베이스 기반 동적 핵심 단어 누락 검사
        
        Args:
            answer: 정답 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 패널티 없음, 낮을수록 큰 패널티)
        """
        critical_words = self.get_critical_words_by_answer(answer)
        
        if not critical_words:
            # 테이블이 없거나 데이터가 없으면 기본 로직 사용
            return self._fallback_critical_word_check(answer, user_input)
        
        penalty = 1.0
        missing_words = []
        
        for keyword, penalty_factor in critical_words:
            if keyword in answer and keyword not in user_input:
                penalty *= penalty_factor
                missing_words.append(keyword)
                logger.info(f"핵심 단어 '{keyword}' 누락 - 패널티: {penalty_factor}")
        
        if missing_words:
            logger.info(f"누락된 핵심 단어: {missing_words}, 최종 패널티: {penalty}")
        
        return penalty
    
    def _fallback_critical_word_check(self, answer: str, user_input: str) -> float:
        """
        기본 핵심 단어 검사 로직 (AI 학습 시스템으로 대체됨)
        """
        # AI 학습 시스템이 자동으로 처리하므로 기본값 반환
        return 1.0
    
    def get_proverb_statistics(self) -> Dict:
        """핵심 단어 통계 조회"""
        conn = self.get_db_connection()
        if not conn:
            return {}
            
        try:
            cursor = conn.cursor(dictionary=True)
            
            # 중요도별 통계
            cursor.execute("""
                SELECT 
                    importance_level,
                    COUNT(*) as keyword_count,
                    AVG(penalty_factor) as avg_penalty,
                    MIN(penalty_factor) as min_penalty,
                    MAX(penalty_factor) as max_penalty
                FROM proverb_keywords
                GROUP BY importance_level
                ORDER BY 
                    CASE importance_level 
                        WHEN 'critical' THEN 1 
                        WHEN 'high' THEN 2 
                        WHEN 'medium' THEN 3 
                        WHEN 'low' THEN 4 
                    END
            """)
            
            stats = {
                'importance_stats': cursor.fetchall()
            }
            
            # 전체 통계
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT proverb_id) as covered_proverbs,
                    COUNT(*) as total_keywords,
                    AVG(penalty_factor) as overall_avg_penalty
                FROM proverb_keywords
            """)
            
            stats['overall'] = cursor.fetchone()
            
            return stats
            
        except mysql.connector.Error as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}
        finally:
            if conn:
                cursor.close()
                conn.close()


# 전역 인스턴스 (싱글톤 패턴)
_keyword_checker_instance = None

def get_keyword_checker() -> ProverbKeywordChecker:
    """키워드 체커 싱글톤 인스턴스 반환"""
    global _keyword_checker_instance
    if _keyword_checker_instance is None:
        _keyword_checker_instance = ProverbKeywordChecker()
    return _keyword_checker_instance


# 기존 함수와 호환성을 위한 래퍼 함수
def detect_dynamic_critical_word_missing(answer: str, user_input: str) -> float:
    """
    기존 함수와 호환되는 래퍼 함수
    
    Args:
        answer: 정답 속담
        user_input: 사용자 입력
        
    Returns:
        float: 패널티 계수
    """
    checker = get_keyword_checker()
    return checker.detect_dynamic_critical_word_missing(answer, user_input)

def get_proverbs_from_db() -> List[Tuple]:
    """데이터베이스에서 속담 데이터를 가져오는 함수"""
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root1234',
        database='proverb_game'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM proverb")
    proverbs = cursor.fetchall()  # [(id, question, answer, hint), ...]
    cursor.close()
    conn.close()
    return proverbs

@lru_cache(maxsize=128)
def get_threshold_by_length(answer: str) -> float:
    """
    속담 길이에 따른 통합된 임계값 설정 (캐시 적용)
    
    통합된 임계값 설정을 사용하여 일관성 있는 유사도 검사 제공
    """
    length = len(answer)
    
    for max_length, threshold in THRESHOLD_MAP:
        if length <= max_length:
            return threshold
    
    return 0.80  # 매우 긴 속담 (30글자 이상) - 동의어 허용

# apply_penalty 함수는 제거됨 (실제 게임에서 사용되지 않음)

# 테스트 케이스 생성 함수들은 제거됨 (실제 게임에서 사용되지 않음)

# detect_compound_word_split 함수는 enhanced_similarity_checker.py로 이동됨

# detect_korean_negation_change 함수는 enhanced_similarity_checker.py로 이동됨

# detect_critical_typos 함수는 enhanced_similarity_checker.py로 이동됨

# ===== 하드코딩된 함수들 제거됨 =====
# 기존의 하드코딩된 함수들은 AI 학습 시스템으로 대체되었습니다.
# IntelligentProverbChecker 클래스가 자동으로 다음을 처리합니다:
# - 동물 혼동 감지 (게→개, 소→말 등)
# - 의미적 반대 관계 감지
# - 핵심 단어 누락 감지
# - 문맥적 관련성 분석

# 백업용 함수들은 AI 학습 시스템으로 대체됨


# enhanced_similarity_check 함수는 enhanced_similarity_checker.py로 이동됨


# 모델 캐싱을 위한 전역 변수
_model_cache = None

def _get_model():
    """SentenceTransformer 모델 싱글톤 패턴으로 관리"""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    return _model_cache

def check_proverb(user_input: str, answers: List[str], threshold: float) -> Tuple[bool, str, float]:
    """
    속담 유사도 검사 (AI 학습 기반 지능적 시스템)
    """
    # AI 학습 시스템 우선 사용 (하드코딩 완전 제거)
    if AI_LEARNING_AVAILABLE:
        try:
            # AI 학습 시스템 사용
            checker = IntelligentProverbChecker()
            
            best_match = ""
            best_score = 0.0
            
            for answer in answers:
                # 기본 유사도 계산
                model = _get_model()
                embeddings = model.encode([answer, user_input])
                base_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                
                # AI 학습 기반 지능적 검사 적용
                enhanced_score = checker.comprehensive_check(answer, user_input, base_similarity)
                
                if enhanced_score > best_score:
                    best_score = enhanced_score
                    best_match = answer
            
            # AI 학습 시스템용 임계값 적용
            ai_threshold = get_threshold_by_length(best_match)
            is_correct = best_score >= ai_threshold
            return is_correct, best_match, best_score
            
        except Exception as e:
            print(f"⚠️ AI 학습 시스템 오류, 향상된 시스템 사용: {e}")
    
    # 백업 1: 향상된 시스템 사용
    if ENHANCED_CHECKER_AVAILABLE:
        try:
            checker = EnhancedProverbSimilarityChecker()
            return checker.check_proverb(user_input, answers, threshold)
        except Exception as e:
            print(f"⚠️ 향상된 시스템 오류, 기본 시스템 사용: {e}")
    
    # 백업 2: 기본 시스템 사용
    try:
        model = _get_model()
        answer_embeddings = model.encode(answers)
        user_embedding = model.encode([user_input])[0]
        similarities = util.cos_sim(user_embedding, answer_embeddings)[0]
        max_idx = similarities.argmax()
        base_score = similarities[max_idx].item()
        matched_answer = answers[max_idx]
        
        # 기본 시스템용 임계값 적용 (더 엄격하게)
        basic_threshold = get_threshold_by_length(matched_answer) + 0.10
        basic_threshold = min(basic_threshold, 0.98)  # 최대 98%로 제한
        return base_score >= basic_threshold, matched_answer, base_score
        
    except Exception as e:
        print(f"❌ 모든 시스템 오류: {e}")
        return False, "", 0.0

def main_game():
    """메인 게임 로직 (최적화)"""
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]  # 미리 답안 리스트 생성
    
    score = 0
    total_time = 60
    start_time = time.time()
    used_idx = set()
    
    print("\n--- 속담 맞추기 게임 시작! ---\n")
    
    while time.time() - start_time < total_time:
        # 중복 문제 방지
        available = [i for i in range(len(proverbs)) if i not in used_idx]
        if not available:
            print("문제를 모두 풀었습니다!")
            break
            
        idx = random.choice(available)
        used_idx.add(idx)
        pid, question, answer, hint = proverbs[idx]
        threshold = get_threshold_by_length(answer)
        
        print(f"문제: {question}")
        wrong_count = 0
        hint_shown = False
        problem_start = time.time()
        
        while True:
            # 제한시간 체크
            if time.time() - start_time >= total_time:
                print("\n제한시간 종료!\n")
                return score
                
            # 10초 경과 시 힌트 자동 표출
            if not hint_shown and time.time() - problem_start >= 10:
                print(f"힌트: {hint}")
                hint_shown = True
                
            user_input = input("속담의 뒷부분(정답)을 입력하세요 (건너뛰기: 엔터): ").strip()
            
            if user_input == "":
                print("다음 문제로 건너뜁니다.\n")
                break
                
            is_correct, matched, score_sim = check_proverb(user_input, answers, threshold)
            
            if is_correct and matched == answer:
                print(f"정답입니다! ({matched}, 유사도: {score_sim:.2f})\n")
                score += 1
                break
            else:
                wrong_count += 1
                print(f"오답입니다. (유사도: {score_sim:.2f})")
                
                if not hint_shown:
                    print(f"힌트: {hint}")
                    hint_shown = True
                    
                if wrong_count >= 2:
                    print(f"정답: {answer}")
                    print("2회 오답! 다음 문제로 넘어갑니다.\n")
                    break
    
    return score

if __name__ == "__main__":
    final_score = main_game()
    print(f"\n게임 종료! 총 점수: {final_score}")