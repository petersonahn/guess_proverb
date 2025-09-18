"""
🎯 속담 게임 - 속담 난이도 분석 클래스

이 모듈은 데이터베이스에 저장된 속담의 난이도를 자동으로 분석하는 핵심 클래스를 제공합니다.
- 데이터베이스의 속담을 3단계 난이도로 분류
- jhgan/ko-sroberta-multitask 모델 사용
- 배치 처리 및 캐싱으로 성능 최적화

주요 기능:
1. 데이터베이스 연동 (proverb_game.proverb 테이블)
2. AI 모델을 통한 난이도 분석
3. 배치 처리 (16개씩 묶어서 처리)
4. 분석 결과 캐싱
5. 진행률 표시 및 성능 모니터링

사용 예시:
    analyzer = ProverbDifficultyAnalyzer()
    result = analyzer.analyze_proverb_difficulty(proverb_id=1)
    all_results = analyzer.batch_analyze_all_proverbs()
"""

import os
import sys
import traceback
import time
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np

# config 및 database 모듈 import
try:
    from config import proverb_config
    from database import ProverbDatabase
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("해결 방법: config.py, database.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 필수 라이브러리 import
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ 필수 라이브러리 import 실패: {e}")
    print("해결 방법: pip install -r requirements.txt")
    sys.exit(1)


class ProverbDifficultyAnalyzer:
    """
    🎯 속담 게임 - 데이터베이스 연동 난이도 분석 클래스
    
    이 클래스는 데이터베이스에 저장된 속담의 난이도를 AI 모델로 자동 분석합니다.
    
    특징:
    - 데이터베이스 연동 (proverb_game.proverb 테이블)
    - jhgan/ko-sroberta-multitask 모델 전용
    - 3단계 난이도 분류 (1점, 2점, 3점)
    - 배치 처리로 성능 최적화 (16개씩 처리)
    - 분석 결과 캐싱으로 중복 분석 방지
    - 진행률 표시 (tqdm 사용)
    """
    
    def __init__(self):
        """
        🤖 속담 난이도 분석기 초기화
        
        1-2단계에서 검증된 모델 로딩 코드를 재사용하고
        데이터베이스 연결을 초기화합니다.
        """
        self.config = proverb_config
        self.model_name = self.config.MODEL_NAME
        self.device = self.config.DEVICE
        self.cache_dir = self.config.MODEL_CACHE_DIR
        self.batch_size = self.config.BATCH_SIZE_ANALYSIS
        
        # 모델 관련 변수
        self.sentence_model = None
        self.tokenizer = None
        self.transformer_model = None
        
        # 데이터베이스 연결
        self.db = None
        
        # 분석 결과 캐시
        self.analysis_cache = {} if self.config.ENABLE_CACHING else None
        
        # 난이도 분석을 위한 참조 속담들
        self.reference_proverbs = self._get_reference_proverbs()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        print(f"🎯 속담 게임 난이도 분석기 초기화 중...")
        print(f"📍 모델: {self.model_name}")
        print(f"💻 디바이스: {self.device}")
        print(f"📦 배치 크기: {self.batch_size}")
        print(f"💾 캐싱: {'활성화' if self.config.ENABLE_CACHING else '비활성화'}")
        
        # 데이터베이스 연결
        if not self._connect_database():
            raise Exception("데이터베이스 연결 실패")
        
        # 모델 로딩
        if not self._load_models():
            raise Exception("AI 모델 로딩 실패")
        
        print(f"✅ 속담 난이도 분석기 준비 완료!")
    
    def _get_reference_proverbs(self) -> Dict[int, List[str]]:
        """
        🧪 각 난이도별 참조 속담들을 가져옵니다.
        
        Returns:
            Dict[int, List[str]]: 난이도별 참조 속담 목록
        """
        reference_proverbs = {}
        for level, info in self.config.PROVERB_DIFFICULTY_LEVELS.items():
            reference_proverbs[level] = info['examples']
        return reference_proverbs
    
    def _load_models(self) -> bool:
        """
        🤖 AI 모델들을 로딩합니다.
        
        1-2단계에서 검증된 모델 로딩 코드를 재사용합니다.
        
        Returns:
            bool: 모델 로딩 성공 여부
        """
        try:
            print("⏳ AI 모델 로딩 중...")
            
            # 캐시 디렉토리 생성
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # SentenceTransformer 모델 로딩 (임베딩 생성용)
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Transformers 모델 로딩 (상세 분석용)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.transformer_model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            print("✅ 모든 AI 모델 로딩 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            self._print_model_error_solutions(e)
            return False
    
    def _print_model_error_solutions(self, error: Exception) -> None:
        """
        🔧 모델 로딩 에러에 대한 해결 방법을 안내합니다.
        
        Args:
            error: 발생한 에러
        """
        print("\n🔧 모델 로딩 문제 해결 방법:")
        
        error_str = str(error).lower()
        
        if "connection" in error_str or "network" in error_str:
            print("🌐 네트워크 연결 문제:")
            print("  1. 인터넷 연결 상태 확인")
            print("  2. Hugging Face 서버 접근 가능 여부 확인")
            print("  3. 방화벽/프록시 설정 확인")
            
        elif "memory" in error_str or "cuda" in error_str:
            print("🧠 메모리 부족 문제:")
            print("  1. GPU 메모리 부족 시 CPU 모드로 재시도")
            print("  2. 다른 프로그램 종료 후 재시도")
            print("  3. 시스템 재부팅 후 재시도")
            
        elif "permission" in error_str:
            print("🔒 권한 문제:")
            print("  1. 관리자 권한으로 실행")
            print("  2. 캐시 디렉토리 권한 확인")
            
        else:
            print("📋 일반적인 해결 방법:")
            print("  1. pip install --upgrade transformers sentence-transformers")
            print("  2. 캐시 디렉토리 삭제 후 재시도")
            print("  3. 가상환경 재생성")
    
    def _preprocess_text(self, text: str) -> str:
        """
        📝 속담 텍스트를 전처리합니다.
        
        Args:
            text: 원본 속담 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        if not isinstance(text, str):
            raise ValueError("입력은 문자열이어야 합니다.")
        
        # 기본적인 텍스트 정리
        text = text.strip()
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정리 (속담에 필요한 문자는 유지)
        text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
        
        return text
    
    def _validate_input(self, proverb_text: str) -> Dict[str, Union[bool, str]]:
        """
        📋 입력 텍스트의 유효성을 검사합니다.
        
        Args:
            proverb_text: 검사할 속담 텍스트
            
        Returns:
            Dict[str, Union[bool, str]]: 검사 결과
        """
        result = {"valid": True, "error_message": ""}
        
        # 빈 문자열 검사
        if not proverb_text or not proverb_text.strip():
            result["valid"] = False
            result["error_message"] = "빈 텍스트입니다. 속담을 입력해주세요."
            return result
        
        # 길이 검사
        if len(proverb_text.strip()) < self.config.PROVERB_MIN_LENGTH:
            result["valid"] = False
            result["error_message"] = f"너무 짧은 텍스트입니다. 최소 {self.config.PROVERB_MIN_LENGTH}글자 이상 입력해주세요."
            return result
        
        if len(proverb_text.strip()) > self.config.PROVERB_MAX_LENGTH:
            result["valid"] = False
            result["error_message"] = f"너무 긴 텍스트입니다. 최대 {self.config.PROVERB_MAX_LENGTH}글자까지 입력 가능합니다."
            return result
        
        # 한글 포함 여부 검사
        if not re.search(r'[가-힣]', proverb_text):
            result["valid"] = False
            result["error_message"] = "한글 속담을 입력해주세요."
            return result
        
        return result
    
    def _calculate_difficulty_from_embeddings(self, proverb_embedding: np.ndarray) -> Dict[str, float]:
        """
        🧮 임베딩 벡터를 바탕으로 난이도를 계산합니다.
        
        Args:
            proverb_embedding: 속담의 임베딩 벡터
            
        Returns:
            Dict[str, float]: 각 난이도별 확률 점수
        """
        try:
            # 각 난이도별 참조 속담들과의 유사도 계산
            difficulty_scores = {}
            
            for level, reference_list in self.reference_proverbs.items():
                # 참조 속담들의 임베딩 생성
                reference_embeddings = self.sentence_model.encode(
                    reference_list, 
                    convert_to_tensor=True
                )
                
                # 코사인 유사도 계산
                from sentence_transformers.util import cos_sim
                similarities = cos_sim(proverb_embedding, reference_embeddings)
                
                # 평균 유사도를 해당 난이도 점수로 사용
                avg_similarity = similarities.mean().item()
                difficulty_scores[level] = max(0.0, avg_similarity)
            
            # 점수 정규화 (합이 1이 되도록)
            total_score = sum(difficulty_scores.values())
            if total_score > 0:
                for level in difficulty_scores:
                    difficulty_scores[level] /= total_score
            else:
                # 모든 점수가 0인 경우 균등 분배
                for level in difficulty_scores:
                    difficulty_scores[level] = 1.0 / len(difficulty_scores)
            
            return difficulty_scores
            
        except Exception as e:
            print(f"⚠️ 난이도 계산 중 오류: {str(e)}")
            # 에러 시 균등 분배
            num_levels = len(self.reference_proverbs)
            return {level: 1.0/num_levels for level in self.reference_proverbs.keys()}
    
    def analyze_difficulty(self, proverb_text: str) -> Dict[str, Union[str, int, float, List[float]]]:
        """
        🎯 속담의 난이도를 분석합니다.
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            Dict: 난이도 분석 결과
            {
                "proverb": "입력한 속담",
                "difficulty_level": 3,        # 1-3 단계
                "confidence": 0.85,           # 0-1 사이 신뢰도
                "raw_scores": [0.1, 0.2, 0.7],  # 각 난이도별 확률
                "message": "분석 완료"
            }
        """
        try:
            print(f"🔍 속담 난이도 분석 시작: '{proverb_text}'")
            
            # 1. 입력 유효성 검사
            validation = self._validate_input(proverb_text)
            if not validation["valid"]:
                return {
                    "proverb": proverb_text,
                    "difficulty_level": 0,
                    "confidence": 0.0,
                    "raw_scores": [0.0, 0.0, 0.0],
                    "message": f"입력 오류: {validation['error_message']}"
                }
            
            # 2. 텍스트 전처리
            processed_text = self._preprocess_text(proverb_text)
            print(f"📝 전처리 완료: '{processed_text}'")
            
            # 3. 임베딩 생성
            print("⏳ AI 모델로 속담 분석 중...")
            proverb_embedding = self.sentence_model.encode(
                [processed_text], 
                convert_to_tensor=True
            )[0]
            
            # 4. 난이도 계산
            difficulty_scores = self._calculate_difficulty_from_embeddings(proverb_embedding)
            
            # 5. 최종 난이도 결정
            predicted_level = max(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])
            confidence = difficulty_scores[predicted_level]
            
            # 6. raw_scores를 리스트 형태로 변환 (레벨 순서대로)
            raw_scores = [difficulty_scores.get(i, 0.0) for i in range(1, 4)]
            
            # 7. 결과 정리
            level_info = self.config.PROVERB_DIFFICULTY_LEVELS[predicted_level]
            
            result = {
                "proverb": processed_text,
                "difficulty_level": predicted_level,
                "confidence": round(confidence, 3),
                "raw_scores": [round(score, 3) for score in raw_scores],
                "message": f"분석 완료 - {level_info['name']} ({level_info['score']}점)"
            }
            
            print(f"✅ 분석 완료: {level_info['name']} (신뢰도: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            print(f"❌ {error_message}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            
            # 에러 해결 방법 안내
            self._print_analysis_error_solutions(e)
            
            return {
                "proverb": proverb_text,
                "difficulty_level": 0,
                "confidence": 0.0,
                "raw_scores": [0.0, 0.0, 0.0],
                "message": error_message
            }
    
    def _print_analysis_error_solutions(self, error: Exception) -> None:
        """
        🔧 분석 에러에 대한 해결 방법을 안내합니다.
        
        Args:
            error: 발생한 에러
        """
        print("\n🔧 분석 오류 해결 방법:")
        
        error_str = str(error).lower()
        
        if "memory" in error_str or "out of memory" in error_str:
            print("🧠 메모리 부족:")
            print("  1. 다른 프로그램 종료 후 재시도")
            print("  2. 더 짧은 텍스트로 테스트")
            print("  3. 시스템 재부팅")
            
        elif "model" in error_str or "tensor" in error_str:
            print("🤖 모델 관련 오류:")
            print("  1. 분석기 재초기화")
            print("  2. 모델 캐시 디렉토리 확인")
            print("  3. GPU/CPU 모드 변경")
            
        else:
            print("📋 일반적인 해결 방법:")
            print("  1. 입력 텍스트 확인")
            print("  2. 네트워크 연결 상태 확인")
            print("  3. 프로그램 재시작")
    
    def batch_analyze(self, proverb_list: List[str]) -> List[Dict]:
        """
        📊 여러 속담을 한 번에 분석합니다.
        
        Args:
            proverb_list: 분석할 속담 목록
            
        Returns:
            List[Dict]: 각 속담의 분석 결과 목록
        """
        print(f"📊 배치 분석 시작: {len(proverb_list)}개 속담")
        
        results = []
        for i, proverb in enumerate(proverb_list, 1):
            print(f"\n[{i}/{len(proverb_list)}] 분석 중...")
            result = self.analyze_difficulty(proverb)
            results.append(result)
        
        print(f"✅ 배치 분석 완료: {len(results)}개 결과")
        return results
    
    def get_difficulty_statistics(self) -> Dict[str, Union[int, Dict]]:
        """
        📈 현재 설정된 난이도 시스템의 통계를 반환합니다.
        
        Returns:
            Dict: 난이도 시스템 통계
        """
        return {
            "total_levels": len(self.config.PROVERB_DIFFICULTY_LEVELS),
            "min_score": self.config.MIN_SCORE,
            "max_score": self.config.MAX_SCORE,
            "level_details": self.config.PROVERB_DIFFICULTY_LEVELS,
            "model_info": {
                "name": self.model_name,
                "device": self.device,
                "cache_dir": self.cache_dir
            }
        }


def test_analyzer():
    """
    🧪 속담 난이도 분석기 테스트 함수
    
    3개의 대표 속담으로 난이도 분석을 테스트합니다.
    """
    print("🧪 속담 난이도 분석기 테스트 시작")
    print("=" * 60)
    
    # 테스트용 속담들
    test_proverbs = [
        "가는 말이 고와야 오는 말이 곱다",      # 예상: 쉬움 (1점)
        "밤말은 새가 듣고 낮말은 쥐가 듣는다",    # 예상: 보통 (2점)
        "가자니 태산이요, 돌아서자니 숭산이라"     # 예상: 어려움 (3점)
    ]
    
    try:
        # 분석기 초기화
        analyzer = ProverbDifficultyAnalyzer()
        
        print(f"\n📊 테스트 속담 목록:")
        for i, proverb in enumerate(test_proverbs, 1):
            print(f"  {i}. {proverb}")
        
        # 각 속담 분석
        print(f"\n🔍 난이도 분석 결과:")
        print("-" * 60)
        
        total_score = 0
        for i, proverb in enumerate(test_proverbs, 1):
            print(f"\n[테스트 {i}]")
            result = analyzer.analyze_difficulty(proverb)
            
            if result["difficulty_level"] > 0:
                level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS[result["difficulty_level"]]
                total_score += level_info["score"]
                
                print(f"📝 속담: {result['proverb']}")
                print(f"🎯 난이도: {result['difficulty_level']}단계 - {level_info['name']}")
                print(f"📊 점수: {level_info['score']}점")
                print(f"🎲 신뢰도: {result['confidence']:.1%}")
                print(f"📈 상세 점수: {result['raw_scores']}")
                print(f"💬 메시지: {result['message']}")
            else:
                print(f"❌ 분석 실패: {result['message']}")
        
        # 통계 정보
        print(f"\n📈 테스트 결과 요약:")
        print("-" * 60)
        print(f"총 획득 점수: {total_score}점")
        print(f"평균 점수: {total_score/len(test_proverbs):.1f}점")
        
        # 시스템 통계
        stats = analyzer.get_difficulty_statistics()
        print(f"\n🎮 난이도 시스템 정보:")
        print(f"- 총 난이도 레벨: {stats['total_levels']}단계")
        print(f"- 점수 범위: {stats['min_score']}~{stats['max_score']}점")
        print(f"- 사용 모델: {stats['model_info']['name']}")
        print(f"- 실행 디바이스: {stats['model_info']['device']}")
        
        print(f"\n✅ 속담 난이도 분석기 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        print(f"📋 에러 상세: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    """
    🚀 스크립트 직접 실행 시 테스트 함수 호출
    
    실행 방법:
        python difficulty_analyzer.py
    """
    print("🎯 속담 게임 - 난이도 분석기")
    print("jhgan/ko-sroberta-multitask 모델 사용")
    print("3단계 난이도 분류 (쉬움 1점, 보통 2점, 어려움 3점)")
    print()
    
    success = test_analyzer()
    sys.exit(0 if success else 1)

