"""
🎯 속담 게임 - 하이브리드 속담 난이도 분석 클래스 (v1.2)

README v1.2에 따른 혁신적인 난이도 분석 시스템:
- 하이브리드 분석: 언어학적 특성(40%) + AI 모델 분석(60%)
- 사용 빈도 중심: 일상적 사용 빈도를 기준으로 한 정확한 난이도 측정
- 패러다임 전환: 속담끼리 비교 → 속담 vs 일반 한국어 사용 패턴 비교
- jhgan/ko-sroberta-multitask 모델 활용한 고도의 난이도 분석

주요 기능:
1. 데이터베이스 연동 (proverb_game.proverb 테이블)
2. AI 기반 사용 빈도 분석 (일상/교육/미디어 문맥과의 유사도)
3. 언어학적 복잡성 분석 (어휘, 구조, 통계적 특성)
4. 3단계 점수 시스템 (1점/2점/3점)
5. 고성능 캐싱 및 배치 처리
6. 실시간 성능 모니터링

사용 예시:
    analyzer = ProverbDifficultyAnalyzer()
    result = analyzer.analyze_proverb_difficulty(proverb_id=1)
    all_results = analyzer.batch_analyze_all_proverbs()
"""

import os
import sys
import time
import traceback
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import numpy as np

# config 및 database 모듈 import
try:
    import sys
    import os
    # 현재 파일의 부모 디렉토리들을 sys.path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # app 디렉토리
    root_dir = os.path.dirname(parent_dir)     # 프로젝트 루트
    sys.path.insert(0, root_dir)
    
    from app.core.config import proverb_config
    from app.includes.dbconn import ProverbDatabase
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("해결 방법: config.py, dbconn.py 파일 경로를 확인하세요.")
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
    🎯 속담 게임 - 하이브리드 난이도 분석 클래스 (v1.2)
    
    README v1.2에 따른 혁신적인 난이도 분석 시스템:
    - 하이브리드 분석: 언어학적 특성(40%) + AI 모델 분석(60%)
    - 사용 빈도 중심: 일상적 사용 빈도를 기준으로 한 정확한 난이도 측정
    - 패러다임 전환: 속담끼리 비교 → 속담 vs 일반 한국어 사용 패턴 비교
    
    특징:
    - 데이터베이스 연동 (proverb_game.proverb 테이블)
    - jhgan/ko-sroberta-multitask 모델 전용
    - AI 기반 사용 빈도 분석 (일상/교육/미디어 문맥)
    - 3단계 난이도 분류 (1점, 2점, 3점)
    - 고성능 캐싱 및 배치 처리
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
        
        # 성능 통계
        self.stats = {
            "total_analyzed": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
            "batch_count": 0
        }
        
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
    
    def _connect_database(self) -> bool:
        """
        🗄️ 데이터베이스 연결을 설정합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            print("⏳ 데이터베이스 연결 중...")
            self.db = ProverbDatabase()
            
            # 연결 테스트
            if not self.db.test_connection():
                print("❌ 데이터베이스 연결 테스트 실패")
                return False
            
            # 테이블 존재 확인
            if not self.db.check_table_exists():
                print("❌ proverb 테이블이 존재하지 않습니다")
                return False
            
            proverb_count = self.db.get_proverb_count()
            print(f"✅ 데이터베이스 연결 성공! (속담 {proverb_count}개)")
            return True
            
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {str(e)}")
            return False
    
    def _load_models(self) -> bool:
        """
        🤖 AI 모델들을 로딩합니다.
        
        Returns:
            bool: 모델 로딩 성공 여부
        """
        try:
            print("⏳ AI 모델 로딩 중...")
            
            # 캐시 디렉토리 생성
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # SentenceTransformer 모델 로딩
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Transformers 모델 로딩 (호환성용)
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
            return False
    
    def analyze_embedding_characteristics(self, proverb_text: str) -> Dict[str, float]:
        """
        🧠 AI 모델 임베딩의 특성을 분석하여 난이도를 추정합니다.
        
        참조 속담과의 유사도 대신, 임베딩 벡터 자체의 특성을 분석합니다.
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            Dict[str, float]: 임베딩 기반 난이도 특성
        """
        try:
            # 임베딩 생성
            embedding = self.sentence_model.encode([proverb_text], convert_to_tensor=True)[0]
            embedding_np = embedding.cpu().numpy()
            
            # 정규화 계수들을 실제 데이터에 맞게 조정
            
            # 1. 벡터 크기 (norm) - 의미적 밀도 (더 보수적으로)
            vector_norm = float(np.linalg.norm(embedding_np))
            # 일반적으로 5-15 범위이므로 조정
            norm_score = max(0.0, min((vector_norm - 5.0) / 10.0, 1.0))
            
            # 2. 벡터 분산 - 의미적 복잡성 (더 보수적으로)
            variance = float(np.var(embedding_np))
            # 일반적으로 0.01-0.1 범위이므로 조정
            variance_score = max(0.0, min((variance - 0.01) / 0.09, 1.0))
            
            # 3. 양수/음수 비율 - 극성 분포 (그대로 유지)
            positive_ratio = float(np.sum(embedding_np > 0) / len(embedding_np))
            polarity_score = abs(positive_ratio - 0.5) * 2  # 0.5에서 멀수록 복잡
            
            # 4. 절댓값 평균 - 활성화 강도 (더 보수적으로)
            abs_mean = float(np.mean(np.abs(embedding_np)))
            # 일반적으로 0.1-0.3 범위이므로 조정
            activation_score = max(0.0, min((abs_mean - 0.1) / 0.2, 1.0))
            
            # 5. 최대값과 최소값의 차이 - 동적 범위 (더 보수적으로)
            dynamic_range = float(np.max(embedding_np) - np.min(embedding_np))
            # 일반적으로 1-4 범위이므로 조정
            range_score = max(0.0, min((dynamic_range - 1.0) / 3.0, 1.0))
            
            # 🔧 추가: 텍스트 길이에 따른 보정
            text_length = len(proverb_text.replace(' ', ''))
            length_factor = min(text_length / 15.0, 1.0)  # 길수록 복잡
            
            # 최종 점수에 길이 보정 적용
            return {
                "semantic_density": norm_score * 0.8 + length_factor * 0.2,      # 의미적 밀도
                "complexity": variance_score * 0.7 + length_factor * 0.3,        # 복잡성
                "polarity": polarity_score,                                       # 극성
                "activation": activation_score * 0.8 + length_factor * 0.2,      # 활성화 강도
                "dynamic_range": range_score * 0.9 + length_factor * 0.1         # 동적 범위
            }
            
        except Exception as e:
            print(f"⚠️ 임베딩 특성 분석 오류: {str(e)}")
            return {
                "semantic_density": 0.5,
                "complexity": 0.5,
                "polarity": 0.5,
                "activation": 0.5,
                "dynamic_range": 0.5
            }
    
    def calculate_linguistic_complexity(self, proverb_text: str) -> float:
        """
        📝 언어적 복잡성 점수 계산 (0.0 ~ 1.0)
        
        Args:
            proverb_text: 분석할 속담 텍스트
        
        Returns:
            float: 언어적 복잡성 점수
            - 0.0: 매우 간단 (레벨 1에 적합)
            - 0.5: 보통 복잡 (레벨 2에 적합)
            - 1.0: 매우 복잡 (레벨 3에 적합)
        """
        try:
            clean_text = proverb_text.replace(' ', '')
            words = proverb_text.split()
            
            # 1. 글자 수 기반 복잡성 (0.0 ~ 1.0)
            # 5글자 이하: 0.0, 15글자 이상: 1.0
            char_complexity = min(max(len(clean_text) - 5, 0) / 10.0, 1.0)
            
            # 2. 단어 수 기반 복잡성 (0.0 ~ 1.0)
            # 3단어 이하: 0.0, 8단어 이상: 1.0
            word_complexity = min(max(len(words) - 3, 0) / 5.0, 1.0)
            
            # 3. 평균 단어 길이 복잡성 (0.0 ~ 1.0)
            # 평균 2글자: 0.0, 평균 4글자 이상: 1.0
            avg_word_len = len(clean_text) / max(len(words), 1)
            word_len_complexity = min(max(avg_word_len - 2, 0) / 2.0, 1.0)
            
            # 4. 음성학적 복잡성 (받침이 많을수록 복잡)
            final_consonants = 0
            for char in clean_text:
                # 한글 받침 검사
                if '가' <= char <= '힣':
                    char_code = ord(char) - ord('가')
                    final_consonant = char_code % 28
                    if final_consonant > 0:  # 받침이 있음
                        final_consonants += 1
            
            consonant_ratio = final_consonants / max(len(clean_text), 1)
            phonetic_complexity = min(consonant_ratio * 2, 1.0)
            
            # 가중 평균으로 최종 복잡성 계산
            final_complexity = (
                char_complexity * 0.3 +
                word_complexity * 0.3 +
                word_len_complexity * 0.2 +
                phonetic_complexity * 0.2
            )
            
            return round(min(max(final_complexity, 0.0), 1.0), 3)
            
        except Exception as e:
            print(f"⚠️ 언어적 복잡성 계산 오류: {str(e)}")
            return 0.5
    
    def calculate_vocabulary_difficulty(self, proverb_text: str) -> float:
        """
        📚 어휘 난이도 점수 계산 (0.0 ~ 1.0)
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 어휘 난이도 점수 (높을수록 어려운 어휘)
        """
        try:
            # 쉬운 일상 어휘들
            easy_vocab = [
                '말', '사람', '물', '집', '길', '손', '눈', '마음', '돈', '밥',
                '하다', '되다', '있다', '없다', '좋다', '나쁘다', '크다', '작다',
                '높다', '낮다', '빠르다', '느리다', '많다', '적다', '새다', '늙다',
                '가다', '오다', '보다', '듣다', '먹다', '자다', '일어나다'
            ]
            
            # 복잡한 어휘 패턴 (한자어, 고어, 전문용어)
            complex_patterns = [
                # 한자어
                '금강산', '태산', '숭산', '백지장', '사공', '원숭이', '식후경',
                # 고어/전통어
                '고와야', '곱다', '편이라', '저리다', '맞들면', '낙이', '석자',
                # 복합어/전문어
                '뿌리기', '떨어진다', '저리다', '맑아야', '어둡다'
            ]
            
            # 현대적 표현
            modern_patterns = ['하면', '되면', '있으면', '없으면', '때문에', '그래서']
            
            words = proverb_text.split()
            total_words = len(words)
            
            if total_words == 0:
                return 0.5
            
            # 쉬운 어휘 비율 계산
            easy_count = sum(1 for word in words if any(easy in word for easy in easy_vocab))
            easy_ratio = easy_count / total_words
            
            # 복잡한 어휘 비율 계산
            complex_count = sum(1 for pattern in complex_patterns if pattern in proverb_text)
            complex_ratio = min(complex_count / total_words, 1.0)
            
            # 현대적 표현 비율
            modern_count = sum(1 for pattern in modern_patterns if pattern in proverb_text)
            modern_ratio = min(modern_count / total_words, 1.0)
            
            # 한자어 추정 (음성학적 특성 기반)
            hanja_like_count = 0
            for word in words:
                if len(word) >= 2:
                    # 한자어는 보통 모음 비율이 낮고 'ㅏㅓㅗㅜ' 같은 기본 모음이 적음
                    basic_vowels = sum(1 for char in word if char in 'ㅏㅓㅗㅜㅡㅣ')
                    if len(word) > 0 and basic_vowels / len(word) < 0.5:  # 기본 모음 비율이 낮으면 한자어 가능성
                        hanja_like_count += 1
            
            hanja_ratio = hanja_like_count / max(total_words, 1)
            
            # 최종 어휘 난이도 계산
            vocab_difficulty = (
                complex_ratio * 0.4 +           # 복잡한 어휘 비율
                hanja_ratio * 0.3 +             # 한자어 추정 비율
                (1 - easy_ratio) * 0.2 +        # 쉬운 어휘 부족
                (1 - modern_ratio) * 0.1        # 현대적 표현 부족
            )
            
            return round(min(max(vocab_difficulty, 0.0), 1.0), 3)
            
        except Exception as e:
            print(f"⚠️ 어휘 난이도 계산 오류: {str(e)}")
            return 0.5
    
    def calculate_structural_simplicity(self, proverb_text: str) -> float:
        """
        🏗️ 구조적 단순성 점수 계산 (0.0 ~ 1.0)
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 구조적 단순성 점수 (높을수록 단순함)
        """
        try:
            # 복잡한 문법 구조 감지
            complex_grammar = [
                '면', '하면', '니', '지만', '어야', '면서',  # 조건문/연결어미
                '이요', '라', '로다', '구나', '이라',        # 고어 종결어미
                '에서', '으로', '에게', '부터', '까지'       # 복합 조사
            ]
            
            # 복잡한 구조 패턴 개수
            complex_count = sum(1 for pattern in complex_grammar if pattern in proverb_text)
            
            # 문장 길이 기반 복잡성
            words = proverb_text.split()
            length_complexity = min(len(words) / 8.0, 1.0)  # 8단어 이상이면 복잡
            
            # 대칭성 점수 계산 (속담의 특징 - 대칭적일수록 단순함)
            symmetry_score = self._calculate_symmetry_score(proverb_text)
            
            # 반복 패턴 점수 (반복될수록 단순함)
            repetition_score = self._calculate_repetition_score(proverb_text)
            
            # 최종 단순성 점수 계산
            # 복잡한 구조가 적고, 대칭성과 반복성이 높을수록 단순함
            simplicity = (
                (1 - min(complex_count / 3.0, 1.0)) * 0.4 +  # 복잡한 구조 적음
                (1 - length_complexity) * 0.3 +              # 짧은 길이
                symmetry_score * 0.2 +                       # 대칭성
                repetition_score * 0.1                       # 반복성
            )
            
            return round(min(max(simplicity, 0.0), 1.0), 3)
            
        except Exception as e:
            print(f"⚠️ 구조적 단순성 계산 오류: {str(e)}")
            return 0.5
    
    def _calculate_symmetry_score(self, text: str) -> float:
        """대칭성 점수 계산"""
        words = text.split()
        
        if len(words) <= 2:
            return 0.8  # 짧은 속담은 대칭성 높음
        
        # 단어 수가 짝수인지 (대칭 구조 가능성)
        even_bonus = 0.3 if len(words) % 2 == 0 else 0.0
        
        # 앞뒤 대칭 구조 확인
        mid = len(words) // 2
        front_half = words[:mid]
        back_half = words[mid:] if len(words) % 2 == 0 else words[mid+1:]
        
        # 구조적 유사성 (길이가 비슷한가)
        if front_half and back_half:
            len_similarity = 1 - abs(len(front_half) - len(back_half)) / max(len(front_half), len(back_half))
            structure_bonus = len_similarity * 0.4
        else:
            structure_bonus = 0.0
        
        # 반복되는 단어나 구조
        repeated_words = len(set(words)) < len(words)
        repeat_bonus = 0.3 if repeated_words else 0.0
        
        return min(even_bonus + structure_bonus + repeat_bonus, 1.0)
    
    def _calculate_repetition_score(self, text: str) -> float:
        """반복 패턴 점수 계산"""
        words = text.split()
        
        if len(words) <= 2:
            return 0.5
        
        # 단어 반복
        unique_words = len(set(words))
        repetition_ratio = 1 - (unique_words / len(words))
        
        # 음성적 반복 (운율)
        syllable_patterns = []
        for word in words:
            if len(word) > 0:
                syllable_patterns.append(word[-1])  # 마지막 글자
        
        unique_endings = len(set(syllable_patterns))
        sound_repetition = 1 - (unique_endings / max(len(syllable_patterns), 1))
        
        return (repetition_ratio * 0.6 + sound_repetition * 0.4)
    
    def analyze_usage_frequency_with_ai(self, proverb_text: str) -> float:
        """
        🧠 AI 기반 사용 빈도 분석 (0.0 ~ 1.0)
        
        README v1.2에 따른 패러다임 전환:
        - 기존: 속담끼리 비교 → 새로운: 속담 vs 일반 한국어 사용 패턴 비교
        - 일상/교육/미디어 문맥에서의 친숙도를 AI로 정확하게 분석
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 사용 빈도 점수 (높을수록 자주 사용됨 = 쉬움)
        """
        try:
            # AI 모델이 로드되지 않은 경우 기본값 반환
            if not self.sentence_model:
                return 0.5
            
            # 1. 일상 문맥 패턴들 (15개 패턴)
            daily_contexts = [
                "항상 말하듯이", "누구나 아는 이야기", "흔히 말하는", "자주 듣는 말",
                "일상에서 쓰는", "평범한 이야기", "간단한 말", "쉬운 이야기",
                "많이 쓰는 말", "흔한 이야기", "보통 말", "일반적인 이야기",
                "자주 하는 말", "흔히 하는 이야기", "평소에 말하는"
            ]
            
            # 2. 교육 문맥 패턴들
            education_contexts = [
                "중요한 교훈", "지혜로운 말씀", "가르치는 이야기", "배우는 내용",
                "교육적인 말", "교훈이 되는", "학습하는 내용", "가르침이 되는"
            ]
            
            # 3. 미디어 문맥 패턴들
            media_contexts = [
                "많은 사람들이 말하는", "널리 알려진", "유명한 이야기", "인기 있는 말",
                "화제가 되는", "많이 회자되는", "널리 퍼진", "인기 있는"
            ]
            
            # 4. 일반적인 한국어 문장 구조 패턴들
            general_structures = [
                "이런 말이 있다", "말로는 이렇다", "보통 이렇게 말한다",
                "흔히 하는 말", "자주 듣는 이야기", "많이 하는 말"
            ]
            
            # 모든 참조 패턴들을 하나의 리스트로 결합
            all_reference_patterns = daily_contexts + education_contexts + media_contexts + general_structures
            
            # 속담과 참조 패턴들 간의 유사도 계산
            proverb_embedding = self.sentence_model.encode([proverb_text], convert_to_tensor=True)[0]
            reference_embeddings = self.sentence_model.encode(all_reference_patterns, convert_to_tensor=True)
            
            # 코사인 유사도 계산
            similarities = []
            for ref_embedding in reference_embeddings:
                similarity = torch.cosine_similarity(proverb_embedding.unsqueeze(0), ref_embedding.unsqueeze(0))
                similarities.append(similarity.item())
            
            # 카테고리별 평균 유사도 계산
            daily_similarity = np.mean(similarities[:len(daily_contexts)])
            education_similarity = np.mean(similarities[len(daily_contexts):len(daily_contexts)+len(education_contexts)])
            media_similarity = np.mean(similarities[len(daily_contexts)+len(education_contexts):len(daily_contexts)+len(education_contexts)+len(media_contexts)])
            structure_similarity = np.mean(similarities[len(daily_contexts)+len(education_contexts)+len(media_contexts):])
            
            # 가중 평균으로 최종 사용 빈도 점수 계산
            # 일상 문맥이 가장 중요 (40%), 교육(30%), 미디어(20%), 구조(10%)
            usage_frequency_score = (
                daily_similarity * 0.4 +      # 일상 문맥 (가장 중요)
                education_similarity * 0.3 +   # 교육 문맥
                media_similarity * 0.2 +       # 미디어 문맥
                structure_similarity * 0.1     # 구조적 친숙성
            )
            
            # 점수를 0.0~1.0 범위로 정규화
            normalized_score = max(0.0, min(usage_frequency_score, 1.0))
            
            return round(normalized_score, 3)
            
        except Exception as e:
            print(f"⚠️ AI 기반 사용 빈도 분석 오류: {str(e)}")
            return 0.5
    
    def calculate_final_difficulty(self, proverb_text: str) -> Dict[str, Any]:
        """
        🎯 하이브리드 최종 난이도 계산 (README v1.2 방식)
        
        혁신적인 분석 방식:
        - 언어학적 특성 분석 (40%): 복잡성, 어휘, 구조
        - AI 기반 사용 빈도 분석 (60%): 일상/교육/미디어 문맥과의 유사도
        - 패러다임 전환: 속담끼리 비교 → 속담 vs 일반 한국어 사용 패턴 비교
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            Dict[str, Any]: 상세한 분석 결과
        """
        try:
            # 1. 언어학적 분석 (40%)
            linguistic_complexity = self.calculate_linguistic_complexity(proverb_text)
            vocab_difficulty = self.calculate_vocabulary_difficulty(proverb_text)
            structural_simplicity = self.calculate_structural_simplicity(proverb_text)
            
            # 2. AI 기반 사용 빈도 분석 (60% 가중치의 핵심)
            usage_frequency = self.analyze_usage_frequency_with_ai(proverb_text)
            
            # 언어학적 종합 점수 (40% 가중치)
            linguistic_score = (
                linguistic_complexity * 0.4 +        # 언어적 복잡성
                vocab_difficulty * 0.3 +              # 어휘 난이도
                (1 - structural_simplicity) * 0.2 +   # 구조적 복잡성
                (1 - usage_frequency) * 0.1           # AI 기반 사용 빈도의 역
            )
            
            # AI 모델 임베딩 분석 (60% 가중치의 핵심)
            embedding_characteristics = self.analyze_embedding_characteristics(proverb_text)
            
            # AI 모델 종합 점수 (사용 빈도 중심)
            ai_score = (
                (1 - usage_frequency) * 0.5 +         # AI 기반 사용 빈도 (가장 중요)
                embedding_characteristics["complexity"] * 0.2 +      # 의미적 복잡성
                embedding_characteristics["semantic_density"] * 0.15 + # 의미적 밀도
                embedding_characteristics["polarity"] * 0.1 +        # 극성
                embedding_characteristics["activation"] * 0.05       # 활성화 강도
            )
            
            # 3. 최종 하이브리드 점수 (README v1.2 가중치: 언어학적 40% + AI 60%)
            final_score = linguistic_score * 0.4 + ai_score * 0.6
            
            # 점수를 1-3 레벨로 변환 (최적화된 경계값)
            if final_score <= 0.45:  # 0.35 → 0.45로 상향 조정
                difficulty_level = 1  # 쉬움
                confidence = 1.0 - final_score * 1.8
            elif final_score <= 0.65:
                difficulty_level = 2  # 보통
                confidence = 1.0 - abs(final_score - 0.55) * 2
            else:
                difficulty_level = 3  # 어려움
                confidence = (final_score - 0.65) * 2 + 0.3
            
            confidence = min(max(confidence, 0.3), 1.0)
            
            # 설명 생성 (README v1.2 하이브리드 버전)
            explanation = self._generate_hybrid_explanation(
                difficulty_level, final_score, linguistic_score, ai_score,
                linguistic_complexity, vocab_difficulty, structural_simplicity, usage_frequency,
                embedding_characteristics
            )
            
            return {
                "difficulty_level": difficulty_level,
                "confidence": round(confidence, 3),
                "final_score": round(final_score, 3),
                "explanation": explanation,
                "breakdown": {
                    "linguistic_analysis": {
                        "linguistic_complexity": linguistic_complexity,
                        "vocabulary_difficulty": vocab_difficulty,
                        "structural_simplicity": structural_simplicity,
                        "usage_frequency": usage_frequency,
                        "linguistic_score": round(linguistic_score, 3)
                    },
                    "ai_analysis": {
                        **embedding_characteristics,
                        "ai_score": round(ai_score, 3)
                    }
                },
                "weights": {
                    "linguistic_analysis": 0.4,
                    "ai_model_analysis": 0.6
                }
            }
            
        except Exception as e:
            print(f"⚠️ 하이브리드 난이도 계산 오류: {str(e)}")
            return {
                "difficulty_level": 2,
                "confidence": 0.5,
                "final_score": 0.5,
                "explanation": f"분석 중 오류 발생: {str(e)}",
                "breakdown": {},
                "weights": {}
            }
    
    def _generate_hybrid_explanation(self, level: int, final_score: float, 
                                   linguistic_score: float, ai_score: float,
                                   ling: float, vocab: float, struct: float, freq: float,
                                   embedding_chars: Dict[str, float]) -> str:
        """하이브리드 분석 결과 설명 생성"""
        level_names = {1: "쉬움", 2: "보통", 3: "어려움"}
        level_name = level_names[level]
        
        reasons = []
        
        # 언어학적 요인 분석
        if ling < 0.3:
            reasons.append("단순한 언어 구조")
        elif ling > 0.7:
            reasons.append("복잡한 언어 구조")
            
        if vocab < 0.3:
            reasons.append("일상적 어휘")
        elif vocab > 0.7:
            reasons.append("어려운 어휘")
            
        # AI 모델 분석 요인
        if embedding_chars["complexity"] > 0.7:
            reasons.append("의미적 복잡성 높음")
        elif embedding_chars["complexity"] < 0.3:
            reasons.append("의미적 단순성")
            
        if embedding_chars["semantic_density"] > 0.7:
            reasons.append("의미 밀도 높음")
        elif embedding_chars["semantic_density"] < 0.3:
            reasons.append("의미 밀도 낮음")
        
        # 주요 분석 방식 표시 (README v1.2 방식)
        dominant_analysis = "AI 기반 사용 빈도" if ai_score > linguistic_score else "언어학적 특성"
        
        explanation = f"{level_name} 난이도 (점수: {final_score:.2f}, {dominant_analysis} 분석 우세)"
        if reasons:
            explanation += f" - {', '.join(reasons[:3])}"
            
        return explanation
    
    def analyze_proverb_difficulty(self, proverb_id: int) -> Dict[str, Any]:
        """
        🎯 데이터베이스 속담의 난이도를 분석합니다.
        
        Args:
            proverb_id (int): 데이터베이스 속담 ID
            
        Returns:
            Dict: 난이도 분석 결과
            {
                "proverb_id": 1,
                "full_proverb": "가는 말이 고와야 오는 말이 곱다", 
                "difficulty_level": 2,
                "confidence": 0.85,
                "score": 2,
                "processing_time": 0.15,
                "message": "분석 완료"
            }
        """
        start_time = time.time()
        
        try:
            # 1. 데이터베이스에서 속담 조회
            proverb_data = self.db.get_proverb_by_id(proverb_id)
            if not proverb_data:
                return {
                    "proverb_id": proverb_id,
                    "full_proverb": "",
                    "difficulty_level": 0,
                    "confidence": 0.0,
                    "score": 0,
                    "processing_time": time.time() - start_time,
                    "message": f"속담 ID {proverb_id}를 찾을 수 없습니다"
                }
            
            full_proverb = proverb_data['full_proverb']
            actual_proverb_id = proverb_data['id']
            
            # 2. 캐시 확인
            cache_key = f"id_{actual_proverb_id}"
            
            if self.analysis_cache and cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key].copy()
                cached_result["processing_time"] = time.time() - start_time
                cached_result["message"] += " (캐시됨)"
                self.stats["cache_hits"] += 1
                return cached_result
            
            print(f"🔍 속담 난이도 분석: '{full_proverb}'")
            
            # 3. 언어학적 특성 기반 난이도 계산 (새로운 하이브리드 방식)
            analysis_result = self.calculate_final_difficulty(full_proverb)
            
            # 4. 결과 정리
            processing_time = time.time() - start_time
            level_info = self.config.PROVERB_DIFFICULTY_LEVELS[analysis_result['difficulty_level']]
            
            result = {
                "proverb_id": actual_proverb_id,
                "full_proverb": full_proverb,
                "difficulty_level": analysis_result['difficulty_level'],
                "confidence": analysis_result['confidence'],
                "score": level_info["score"],
                "processing_time": round(processing_time, 3),
                "message": analysis_result['explanation'],
                "analysis_breakdown": analysis_result['breakdown'],
                "final_score": analysis_result['final_score']
            }
            
            # 8. 캐시 저장
            if self.analysis_cache:
                self.analysis_cache[cache_key] = result.copy()
            
            # 9. 통계 업데이트
            self.stats["total_analyzed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            level_name = level_info['name']
            print(f"✅ 분석 완료: {level_name} (신뢰도: {analysis_result['confidence']:.1%}, {processing_time:.3f}초)")
            
            return result
            
        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            print(f"❌ {error_message}")
            
            return {
                "proverb_id": proverb_id,
                "full_proverb": "",
                "difficulty_level": 0,
                "confidence": 0.0,
                "score": 0,
                "processing_time": time.time() - start_time,
                "message": error_message
            }
    
    def batch_analyze_all_proverbs(self) -> List[Dict[str, Any]]:
        """
        📊 데이터베이스의 모든 속담을 배치 처리로 분석합니다.
        
        16개씩 묶어서 처리하여 메모리 효율성을 확보하고,
        진행률을 표시합니다.
        
        Returns:
            List[Dict]: 모든 속담의 분석 결과
        """
        print("📊 전체 속담 배치 분석 시작")
        print("=" * 60)
        
        start_time = time.time()
        all_results = []
        
        try:
            # 전체 속담 개수 조회
            total_proverbs = self.db.get_proverb_count()
            if total_proverbs == 0:
                print("❌ 분석할 속담이 없습니다")
                return []
            
            print(f"📚 총 {total_proverbs}개 속담 분석 예정")
            print(f"📦 배치 크기: {self.batch_size}개")
            
            # 배치 개수 계산
            total_batches = (total_proverbs + self.batch_size - 1) // self.batch_size
            print(f"🔄 총 {total_batches}개 배치로 처리")
            
            # 진행률 바 설정
            with tqdm(total=total_proverbs, desc="속담 분석 진행", unit="개") as pbar:
                
                for batch_idx in range(total_batches):
                    offset = batch_idx * self.batch_size
                    
                    # 배치 데이터 조회
                    batch_proverbs = self.db.get_proverbs_batch(offset, self.batch_size)
                    
                    if not batch_proverbs:
                        break
                    
                    print(f"\n[배치 {batch_idx + 1}/{total_batches}] {len(batch_proverbs)}개 속담 처리 중...")
                    
                    # 배치 내 각 속담 분석
                    batch_results = []
                    for proverb_data in batch_proverbs:
                        result = self.analyze_proverb_difficulty(proverb_id=proverb_data['id'])
                        batch_results.append(result)
                        pbar.update(1)
                    
                    all_results.extend(batch_results)
                    self.stats["batch_count"] += 1
                    
                    # 배치 완료 정보
                    success_count = sum(1 for r in batch_results if r['difficulty_level'] > 0)
                    print(f"✅ 배치 {batch_idx + 1} 완료: {success_count}/{len(batch_results)}개 성공")
            
            # 전체 결과 통계
            total_time = time.time() - start_time
            success_results = [r for r in all_results if r['difficulty_level'] > 0]
            
            print(f"\n" + "=" * 60)
            print(f"📈 배치 분석 완료 결과:")
            print(f"  - 총 처리 시간: {total_time:.2f}초")
            print(f"  - 성공적으로 분석된 속담: {len(success_results)}/{len(all_results)}개")
            print(f"  - 평균 처리 시간: {total_time/len(all_results):.3f}초/개")
            print(f"  - 캐시 적중률: {self.stats['cache_hits']}/{self.stats['total_analyzed']} ({self.stats['cache_hits']/max(1,self.stats['total_analyzed'])*100:.1f}%)")
            
            # 난이도별 분포
            self._print_difficulty_distribution(success_results)
            
            return all_results
            
        except Exception as e:
            print(f"❌ 배치 분석 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return all_results
    
    def _print_difficulty_distribution(self, results: List[Dict[str, Any]]) -> None:
        """
        📊 난이도 분포를 출력합니다.
        
        Args:
            results: 분석 결과 목록
        """
        if not results:
            return
        
        # 난이도별 개수 계산
        difficulty_counts = {1: 0, 2: 0, 3: 0}
        total_score = 0
        
        for result in results:
            level = result.get('difficulty_level', 0)
            if level in difficulty_counts:
                difficulty_counts[level] += 1
                total_score += result.get('score', 0)
        
        total_count = len(results)
        
        print(f"\n📊 난이도 분포:")
        for level, count in difficulty_counts.items():
            level_info = self.config.PROVERB_DIFFICULTY_LEVELS[level]
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            print(f"  - {level_info['name']}: {count}개 ({percentage:.1f}%)")
        
        avg_score = total_score / total_count if total_count > 0 else 0
        print(f"\n🎯 평균 점수: {avg_score:.2f}점")
        print(f"🏆 총 획득 가능 점수: {total_score}점")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        📈 분석 통계 정보를 반환합니다.
        
        Returns:
            Dict: 통계 정보
        """
        return {
            "total_analyzed": self.stats["total_analyzed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_analyzed"]),
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": self.stats["total_processing_time"] / max(1, self.stats["total_analyzed"]),
            "batch_count": self.stats["batch_count"],
            "cache_size": len(self.analysis_cache) if self.analysis_cache else 0,
            "analysis_method": "하이브리드 (언어학적 + AI 모델)",
            "ai_model": self.model_name
        }
    
    def clear_cache(self):
        """💾 분석 결과 캐시를 초기화합니다."""
        if self.analysis_cache:
            cache_size = len(self.analysis_cache)
            self.analysis_cache.clear()
            print(f"🗑️ 캐시 초기화 완료 ({cache_size}개 항목 삭제)")
    
    def close(self):
        """🔒 리소스를 정리하고 연결을 종료합니다."""
        if self.db:
            self.db.close()
        
        # 캐시 정리
        if self.analysis_cache:
            self.analysis_cache.clear()
        
        print("✅ 속담 난이도 분석기 종료")


def test_difficulty_analyzer():
    """
    🧪 속담 난이도 분석기 종합 테스트
    """
    print("🧪 속담 난이도 분석기 종합 테스트 시작")
    print("=" * 60)
    
    try:
        # 분석기 초기화
        analyzer = ProverbDifficultyAnalyzer()
        
        # 1. 개별 속담 분석 테스트
        print(f"\n🔍 개별 속담 분석 테스트:")
        print("-" * 40)
        
        # ID로 분석
        result1 = analyzer.analyze_proverb_difficulty(proverb_id=1)
        if result1['difficulty_level'] > 0:
            print(f"✅ ID 분석 성공:")
            print(f"   속담: {result1['full_proverb']}")
            print(f"   난이도: {result1['difficulty_level']}단계 ({result1['score']}점)")
            print(f"   신뢰도: {result1['confidence']:.1%}")
            print(f"   처리시간: {result1['processing_time']:.3f}초")
        
        # 다른 ID로 분석
        result2 = analyzer.analyze_proverb_difficulty(proverb_id=2)
        if result2['difficulty_level'] > 0:
            print(f"\n✅ 추가 분석 성공:")
            print(f"   속담: {result2['full_proverb']}")
            print(f"   난이도: {result2['difficulty_level']}단계 ({result2['score']}점)")
            print(f"   신뢰도: {result2['confidence']:.1%}")
        
        # 2. 배치 분석 테스트 (처음 5개만)
        print(f"\n📦 배치 분석 테스트 (처음 5개 속담):")
        print("-" * 40)
        
        # 임시로 배치 크기를 5로 설정
        original_batch_size = analyzer.batch_size
        analyzer.batch_size = 5
        
        batch_results = []
        batch_proverbs = analyzer.db.get_proverbs_batch(0, 5)
        
        for proverb_data in batch_proverbs:
            result = analyzer.analyze_proverb_difficulty(proverb_id=proverb_data['id'])
            batch_results.append(result)
        
        # 결과 테이블 형태로 출력
        print(f"\n📋 분석 결과 테이블:")
        print(f"{'ID':<4} {'속담':<30} {'난이도':<8} {'점수':<4} {'신뢰도':<8}")
        print("-" * 60)
        
        for result in batch_results:
            if result['difficulty_level'] > 0:
                proverb_short = result['full_proverb'][:25] + "..." if len(result['full_proverb']) > 25 else result['full_proverb']
                level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS[result['difficulty_level']]
                print(f"{result['proverb_id']:<4} {proverb_short:<30} {level_info['name']:<8} {result['score']:<4} {result['confidence']:.1%}")
        
        # 배치 크기 복원
        analyzer.batch_size = original_batch_size
        
        # 3. 통계 정보 출력
        print(f"\n📈 분석 통계:")
        print("-" * 40)
        stats = analyzer.get_analysis_statistics()
        print(f"총 분석 횟수: {stats['total_analyzed']}개")
        print(f"캐시 적중: {stats['cache_hits']}회 ({stats['cache_hit_rate']:.1%})")
        print(f"평균 처리시간: {stats['average_processing_time']:.3f}초")
        print(f"캐시 크기: {stats['cache_size']}개")
        
        # 4. 정리
        analyzer.close()
        
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
        python difficulty_analyzer_new.py
    """
    print("🎯 속담 게임 - 하이브리드 속담 난이도 분석기 (v1.2)")
    print("🔬 분석 방식: 언어학적 특성(40%) + AI 기반 사용 빈도 분석(60%)")
    print("📊 패러다임 전환: 속담끼리 비교 → 속담 vs 일반 한국어 사용 패턴 비교")
    print("🗄️ 데이터베이스: proverb_game.proverb (root/0000)")
    print()
    
    success = test_difficulty_analyzer()
    sys.exit(0 if success else 1)
