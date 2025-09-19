"""
🎯 속담 게임 - 하이브리드 속담 난이도 분석 클래스

이 모듈은 언어학적 특성과 AI 모델을 결합하여 속담의 난이도를 분석합니다.
- 언어학적 분석 (60%): 복잡성, 어휘, 구조, 빈도
- AI 모델 분석 (40%): models--jhgan--ko-sroberta-multitask 임베딩 특성 분석

주요 기능:
1. 데이터베이스 연동 (proverb_game.proverb 테이블)
2. 하이브리드 난이도 분석 (언어학적 + AI 모델)
3. models--jhgan--ko-sroberta-multitask 모델 전용 사용
4. 배치 처리 (16개씩 묶어서 처리)
5. 분석 결과 캐싱 및 성능 모니터링
6. 진행률 표시 (tqdm)

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
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

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
    import numpy as np
except ImportError as e:
    print(f"❌ 필수 라이브러리 import 실패: {e}")
    print("해결 방법: pip install -r requirements.txt")
    sys.exit(1)


class ProverbDifficultyAnalyzer:
    """
    🎯 속담 게임 - 하이브리드 난이도 분석 클래스
    
    언어학적 특성과 AI 모델을 결합하여 속담의 난이도를 분석합니다.
    
    특징:
    - 데이터베이스 연동 (proverb_game.proverb 테이블)
    - models--jhgan--ko-sroberta-multitask 모델 전용 사용
    - 하이브리드 분석: 언어학적 분석(60%) + AI 임베딩 분석(40%)
    - 3단계 난이도 분류 (1점, 2점, 3점)
    - 배치 처리로 성능 최적화 (16개씩 처리)
    - 분석 결과 캐싱으로 중복 분석 방지
    """
    
    def __init__(self):
        """
        🤖 속담 난이도 분석기 초기화 (하이브리드 분석)
        """
        self.config = proverb_config
        self.model_name = self.config.MODEL_NAME
        self.device = self.config.DEVICE
        self.cache_dir = self.config.MODEL_CACHE_DIR
        self.batch_size = self.config.BATCH_SIZE_ANALYSIS
        
        # AI 모델 초기화
        self.sentence_model = None
        
        # 데이터베이스 연결
        self.db = None
        
        # 분석 결과 캐시
        self.analysis_cache = {} if self.config.ENABLE_CACHING else None
        
        # 성능 통계
        self.stats = {
            "total_analyzed": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
            "batch_count": 0
        }
        
        print(f"🎯 속담 게임 난이도 분석기 초기화 중...")
        print(f"🤖 AI 모델: {self.model_name}")
        print(f"💻 디바이스: {self.device}")
        print(f"🔬 분석 방식: 사용 빈도 중심 (AI 사용빈도 60% + 언어적복잡성 40%)")
        print(f"📦 배치 크기: {self.batch_size}")
        print(f"💾 캐싱: {'활성화' if self.config.ENABLE_CACHING else '비활성화'}")
        
        # AI 모델 로딩
        if not self._load_ai_model():
            raise Exception("AI 모델 로딩 실패")
        
        # 데이터베이스 연결
        if not self._connect_database():
            raise Exception("데이터베이스 연결 실패")
        
        print(f"✅ 속담 난이도 분석기 준비 완료!")
    
    def _load_ai_model(self) -> bool:
        """
        🤖 models--jhgan--ko-sroberta-multitask 모델을 로딩합니다.
        
        Returns:
            bool: 모델 로딩 성공 여부
        """
        try:
            print("⏳ AI 모델 로딩 중...")
            print(f"📍 모델: {self.model_name}")
            print(f"📁 캐시 디렉토리: {self.cache_dir}")
            
            # SentenceTransformer 모델 로딩
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # 모델 테스트
            test_text = "가는 말이 고와야 오는 말이 곱다"
            test_embedding = self.sentence_model.encode([test_text], convert_to_tensor=True)
            
            print(f"✅ AI 모델 로딩 성공!")
            print(f"📊 임베딩 차원: {test_embedding.shape[1]}")
            print(f"💻 모델 디바이스: {test_embedding.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ AI 모델 로딩 실패: {str(e)}")
            return False
    
    def analyze_usage_frequency_with_ai(self, proverb_text: str) -> Dict[str, float]:
        """
        🧠 AI 모델을 활용한 사용 빈도 분석 (새로운 접근법)
        
        속담과 "일반 한국어 사용 패턴"과의 유사도를 측정하여 사용 빈도를 추정합니다.
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            Dict[str, float]: AI 기반 사용 빈도 특성
        """
        try:
            # 속담 임베딩 생성
            proverb_embedding = self.sentence_model.encode([proverb_text], convert_to_tensor=True)[0]
            
            # 속담과 유사한 한국어 표현 패턴들 (속담에 특화)
            common_usage_patterns = [
                # 일상에서 자주 인용되는 표현 패턴
                "항상 말하듯이",
                "누구나 아는 이야기지만",
                "어른들이 말씀하시길",
                "옛날부터 전해오는 말로",
                "우리가 흔히 하는 말이",
                
                # 교육/훈계적 문맥
                "중요한 교훈이 있다면",
                "경험으로 배운 것은",
                "지혜로운 말씀이 있죠",
                "생각해볼 만한 이야기",
                "기억해둘 만한 말씀",
                
                # 일반적 지식/상식 표현
                "많은 사람들이 말하는",
                "일반적으로 알려진",
                "누구나 공감하는",
                "흔히 들어본 이야기",
                "널리 알려진 말씀"
            ]
            
            # 각 패턴과의 유사도 계산
            pattern_embeddings = self.sentence_model.encode(common_usage_patterns, convert_to_tensor=True)
            
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(proverb_embedding, pattern_embeddings)
            
            # 1. 일상 대화와의 유사도 (0-4번 패턴)
            daily_similarity = similarities[0:5].mean().item()
            
            # 2. 교육적 문맥과의 유사도 (5-9번 패턴)  
            educational_similarity = similarities[5:10].mean().item()
            
            # 3. 미디어/문학적 표현과의 유사도 (10-14번 패턴)
            media_similarity = similarities[10:15].mean().item()
            
            # 4. 전체 평균 유사도
            overall_similarity = similarities.mean().item()
            
            # 5. 문장 구조 친숙성 (간단한 구조일수록 친숙함)
            words = proverb_text.split()
            structure_familiarity = max(0.0, min(1.0 - (len(words) - 4) / 8.0, 1.0))  # 4-12단어 기준, 1.0 제한
            
            return {
                "daily_context_similarity": max(0.0, daily_similarity),           # 일상 문맥 유사도
                "educational_similarity": max(0.0, educational_similarity),       # 교육적 유사도
                "media_similarity": max(0.0, media_similarity),                   # 미디어 유사도
                "overall_familiarity": max(0.0, overall_similarity),             # 전체 친숙도
                "structure_familiarity": structure_familiarity                    # 구조적 친숙성
            }
            
        except Exception as e:
            print(f"⚠️ AI 사용 빈도 분석 오류: {str(e)}")
            return {
                "daily_context_similarity": 0.5,
                "educational_similarity": 0.5,
                "media_similarity": 0.5,
                "overall_familiarity": 0.5,
                "structure_familiarity": 0.5
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
        📚 어휘 난이도 점수 계산 (통계적 접근, 하드코딩 최소화)
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 어휘 난이도 점수 (높을수록 어려운 어휘)
        """
        try:
            words = proverb_text.split()
            total_words = len(words)
            
            if total_words == 0:
                return 0.5
            
            # 1. 평균 단어 길이 (긴 단어일수록 어려움)
            avg_word_length = sum(len(word) for word in words) / total_words
            length_complexity = min(max(avg_word_length - 2, 0) / 3.0, 1.0)
            
            # 2. 한글 자모 복잡성 (복잡한 자음/모음 조합)
            complex_chars = 0
            total_chars = 0
            for word in words:
                for char in word:
                    if '가' <= char <= '힣':
                        total_chars += 1
                        # 복잡한 받침이나 복합 모음
                        char_code = ord(char) - ord('가')
                        final_consonant = char_code % 28
                        if final_consonant > 16:  # 복잡한 받침 (ㄺ, ㄻ, ㄼ 등)
                            complex_chars += 1
            
            char_complexity = complex_chars / max(total_chars, 1)
            
            # 3. 어휘 다양성 (고유 단어 비율이 높을수록 복잡)
            unique_words = len(set(words))
            diversity = unique_words / total_words
            
            # 4. 음성학적 복잡성 (어려운 발음 패턴)
            phonetic_complexity = 0
            for word in words:
                # 연속된 자음이나 복잡한 패턴
                if len(word) >= 3:
                    # 3글자 이상 단어에서 복잡한 패턴 감지
                    if any(consonant in word for consonant in ['ㄱㄱ', 'ㄴㄴ', 'ㄹㄹ']):
                        phonetic_complexity += 0.2
            
            phonetic_score = min(phonetic_complexity / total_words, 1.0)
            
            # 최종 어휘 난이도 (통계적 접근)
            vocab_difficulty = (
                length_complexity * 0.4 +      # 단어 길이
                char_complexity * 0.3 +        # 자모 복잡성
                diversity * 0.2 +               # 어휘 다양성
                phonetic_score * 0.1           # 음성학적 복잡성
            )
            
            return round(min(max(vocab_difficulty, 0.0), 1.0), 3)
            
        except Exception as e:
            print(f"⚠️ 어휘 난이도 계산 오류: {str(e)}")
            return 0.5
    
    def calculate_structural_simplicity(self, proverb_text: str) -> float:
        """
        🏗️ 구조적 단순성 점수 계산 (통계적 접근)
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 구조적 단순성 점수 (높을수록 단순함)
        """
        try:
            words = proverb_text.split()
            total_words = len(words)
            
            if total_words == 0:
                return 0.5
            
            # 1. 문장 길이 단순성 (짧을수록 단순)
            length_simplicity = max(0.0, 1.0 - (total_words - 3) / 8.0)
            
            # 2. 어미 복잡성 (복잡한 어미가 적을수록 단순)
            complex_endings = ['면서', '지만', '더라도', '에도', '라도']
            ending_complexity = sum(1 for ending in complex_endings if ending in proverb_text)
            ending_simplicity = max(0.0, 1.0 - ending_complexity / 3.0)
            
            # 3. 대칭성 (속담의 특징 - 대칭적일수록 기억하기 쉬움)
            symmetry_score = self._calculate_symmetry_score(proverb_text)
            
            # 4. 반복성 (반복 패턴이 있을수록 단순)
            repetition_score = self._calculate_repetition_score(proverb_text)
            
            # 5. 구두점/특수문자 복잡성
            punctuation_count = sum(1 for char in proverb_text if char in '.,!?;:')
            punctuation_simplicity = max(0.0, 1.0 - punctuation_count / 3.0)
            
            # 최종 단순성 점수 (통계적 접근)
            simplicity = (
                length_simplicity * 0.3 +        # 길이 단순성
                ending_simplicity * 0.25 +       # 어미 단순성
                symmetry_score * 0.2 +           # 대칭성
                repetition_score * 0.15 +        # 반복성
                punctuation_simplicity * 0.1     # 구두점 단순성
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
    
    def estimate_usage_frequency(self, proverb_text: str) -> float:
        """
        📊 사용 빈도 추정 (통계적 접근, 하드코딩 최소화)
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            float: 사용 빈도 추정값 (높을수록 자주 사용됨 = 쉬움)
        """
        try:
            clean_text = proverb_text.replace(' ', '')
            words = proverb_text.split()
            
            # 1. 길이 기반 빈도 (짧을수록 기억하기 쉬워 자주 사용)
            length_score = max(0.0, 1.0 - (len(clean_text) - 4) / 12.0)
            
            # 2. 단어 단순성 (단순한 단어일수록 자주 사용)
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            word_simplicity = max(0.0, 1.0 - (avg_word_length - 2) / 4.0)
            
            # 3. 음성학적 단순성 (발음하기 쉬울수록 자주 사용)
            phonetic_simplicity = 0
            for word in words:
                if len(word) <= 3:  # 짧은 단어
                    phonetic_simplicity += 0.2
                # 반복 음성 패턴 (기억하기 쉬움)
                if len(set(word)) < len(word):  # 반복되는 글자
                    phonetic_simplicity += 0.1
            
            phonetic_score = min(phonetic_simplicity / len(words), 1.0)
            
            # 4. 리듬감 (운율이 있을수록 기억하기 쉬워 자주 사용)
            rhythm_score = 0
            if len(words) % 2 == 0:  # 짝수 단어 (대칭성)
                rhythm_score += 0.3
            
            # 단어 길이 패턴의 규칙성
            word_lengths = [len(word) for word in words]
            if len(set(word_lengths)) <= 2:  # 비슷한 길이의 단어들
                rhythm_score += 0.2
            
            rhythm_score = min(rhythm_score, 1.0)
            
            # 5. 구조적 친숙성 (일반적인 문장 구조일수록 자주 사용)
            structure_familiarity = 0
            if 3 <= len(words) <= 7:  # 적당한 길이
                structure_familiarity += 0.4
            
            # 반복 구조나 대칭 구조
            if len(words) >= 4:
                mid = len(words) // 2
                if words[:mid] == words[mid:] or len(set(words)) < len(words):
                    structure_familiarity += 0.3
            
            structure_score = min(structure_familiarity, 1.0)
            
            # 최종 사용 빈도 점수 (통계적 접근)
            frequency_score = (
                length_score * 0.3 +           # 길이 단순성
                word_simplicity * 0.25 +       # 단어 단순성
                phonetic_score * 0.2 +         # 음성학적 단순성
                rhythm_score * 0.15 +          # 리듬감
                structure_score * 0.1          # 구조적 친숙성
            )
            
            return round(min(max(frequency_score, 0.0), 1.0), 3)
            
        except Exception as e:
            print(f"⚠️ 사용 빈도 추정 오류: {str(e)}")
            return 0.5
    
    def calculate_final_difficulty(self, proverb_text: str) -> Dict[str, Any]:
        """
        🎯 사용 빈도 중심 난이도 계산 (새로운 접근법)
        
        사용 빈도(60%) + 언어적 복잡성(40%)으로 난이도를 판별합니다.
        
        Args:
            proverb_text: 분석할 속담 텍스트
            
        Returns:
            Dict[str, Any]: 상세한 분석 결과
        """
        try:
            # 1. AI 기반 사용 빈도 분석 (60%) - 주요 지표
            ai_frequency_analysis = self.analyze_usage_frequency_with_ai(proverb_text)
            
            # 사용 빈도 종합 점수 (높을수록 자주 사용됨 = 쉬움)
            usage_frequency_score = (
                ai_frequency_analysis["daily_context_similarity"] * 0.3 +    # 일상 문맥
                ai_frequency_analysis["educational_similarity"] * 0.25 +     # 교육적 문맥
                ai_frequency_analysis["media_similarity"] * 0.2 +            # 미디어 문맥
                ai_frequency_analysis["overall_familiarity"] * 0.15 +        # 전체 친숙도
                ai_frequency_analysis["structure_familiarity"] * 0.1         # 구조적 친숙성
            )
            
            # 2. 언어적 복잡성 분석 (40%) - 보조 지표
            linguistic_complexity = self.calculate_linguistic_complexity(proverb_text)
            vocab_difficulty = self.calculate_vocabulary_difficulty(proverb_text)
            structural_simplicity = self.calculate_structural_simplicity(proverb_text)
            
            # 언어적 복잡성 종합 점수 (높을수록 복잡함 = 어려움)
            complexity_score = (
                linguistic_complexity * 0.4 +          # 언어적 복잡성
                vocab_difficulty * 0.35 +              # 어휘 난이도
                (1 - structural_simplicity) * 0.25     # 구조적 복잡성
            )
            
            # 3. 최종 난이도 점수 계산
            # 사용 빈도가 높을수록 쉽고, 복잡성이 높을수록 어려움
            final_score = (1 - usage_frequency_score) * 0.6 + complexity_score * 0.4
            
            # 점수를 1-3 레벨로 변환 (최적 성능 경계값)
            if final_score <= 0.58:  # 쉬운 속담
                difficulty_level = 1  # 쉬움
                confidence = 1.0 - final_score * 1.2
            elif final_score <= 0.68:  # 중간 범위
                difficulty_level = 2  # 보통
                confidence = 1.0 - abs(final_score - 0.63) * 3
            else:
                difficulty_level = 3  # 어려움
                confidence = (final_score - 0.68) * 3 + 0.4
            
            confidence = min(max(confidence, 0.3), 1.0)
            
            # 설명 생성 (사용 빈도 중심)
            explanation = self._generate_frequency_based_explanation(
                difficulty_level, final_score, usage_frequency_score, complexity_score,
                ai_frequency_analysis
            )
            
            return {
                "difficulty_level": difficulty_level,
                "confidence": round(confidence, 3),
                "final_score": round(final_score, 3),
                "explanation": explanation,
                "breakdown": {
                    "usage_frequency_analysis": {
                        **ai_frequency_analysis,
                        "usage_frequency_score": round(usage_frequency_score, 3)
                    },
                    "linguistic_complexity_analysis": {
                        "linguistic_complexity": linguistic_complexity,
                        "vocabulary_difficulty": vocab_difficulty,
                        "structural_simplicity": structural_simplicity,
                        "complexity_score": round(complexity_score, 3)
                    }
                },
                "weights": {
                    "usage_frequency": 0.6,
                    "linguistic_complexity": 0.4
                }
            }
            
        except Exception as e:
            print(f"⚠️ 사용 빈도 중심 난이도 계산 오류: {str(e)}")
            return {
                "difficulty_level": 2,
                "confidence": 0.5,
                "final_score": 0.5,
                "explanation": f"분석 중 오류 발생: {str(e)}",
                "breakdown": {},
                "weights": {}
            }
    
    def _generate_explanation(self, level: int, score: float, ling: float, 
                            vocab: float, struct: float, freq: float) -> str:
        """분석 결과 설명 생성 (기존 버전)"""
        level_names = {1: "쉬움", 2: "보통", 3: "어려움"}
        level_name = level_names[level]
        
        reasons = []
        
        # 주요 요인 분석
        if ling < 0.3:
            reasons.append("단순한 언어 구조")
        elif ling > 0.7:
            reasons.append("복잡한 언어 구조")
            
        if vocab < 0.3:
            reasons.append("일상적 어휘")
        elif vocab > 0.7:
            reasons.append("어려운 어휘")
            
        if struct > 0.7:
            reasons.append("대칭적 구조")
        elif struct < 0.3:
            reasons.append("복잡한 문법")
            
        if freq > 0.7:
            reasons.append("자주 사용되는 표현")
        elif freq < 0.3:
            reasons.append("잘 알려지지 않은 표현")
        
        explanation = f"{level_name} 난이도 (점수: {score:.2f})"
        if reasons:
            explanation += f" - {', '.join(reasons[:3])}"
            
        return explanation
    
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
        
        # 주요 분석 방식 표시
        dominant_analysis = "언어학적" if linguistic_score > ai_score else "AI 모델"
        
        explanation = f"{level_name} 난이도 (점수: {final_score:.2f}, {dominant_analysis} 분석 우세)"
        if reasons:
            explanation += f" - {', '.join(reasons[:3])}"
            
        return explanation
    
    def _generate_frequency_based_explanation(self, level: int, final_score: float,
                                            usage_frequency_score: float, complexity_score: float,
                                            ai_analysis: Dict[str, float]) -> str:
        """사용 빈도 중심 분석 결과 설명 생성"""
        level_names = {1: "쉬움", 2: "보통", 3: "어려움"}
        level_name = level_names[level]
        
        reasons = []
        
        # 사용 빈도 요인 분석
        if usage_frequency_score > 0.7:
            reasons.append("일상적으로 자주 사용되는 표현")
        elif usage_frequency_score < 0.3:
            reasons.append("잘 알려지지 않은 표현")
        
        # AI 분석 세부 요인
        if ai_analysis["daily_context_similarity"] > 0.6:
            reasons.append("일상 대화에 친숙한 패턴")
        elif ai_analysis["educational_similarity"] > 0.6:
            reasons.append("교육적 맥락에서 친숙함")
        elif ai_analysis["media_similarity"] > 0.6:
            reasons.append("미디어에서 자주 접하는 형태")
        
        # 복잡성 요인 분석
        if complexity_score > 0.7:
            reasons.append("언어적으로 복잡한 구조")
        elif complexity_score < 0.3:
            reasons.append("언어적으로 단순한 구조")
        
        # 주요 분석 방식 표시
        dominant_factor = "사용 빈도" if usage_frequency_score > complexity_score else "언어적 복잡성"
        
        explanation = f"{level_name} 난이도 (점수: {final_score:.2f}, {dominant_factor} 기준)"
        if reasons:
            explanation += f" - {', '.join(reasons[:3])}"
            
        return explanation
    
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
    
    def analyze_proverb_difficulty(self, proverb_id: int) -> Dict[str, Any]:
        """
        🎯 언어학적 특성 기반 속담 난이도 분석 (완전 재설계)
        
        참조 속담과의 의미적 유사도 대신 언어학적 특성만으로 난이도를 판별합니다.
        
        Args:
            proverb_id (int): 데이터베이스 속담 ID
        
        Returns:
            Dict: 난이도 분석 결과
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
            
            # 3. 언어학적 특성 기반 난이도 계산 (새로운 방식)
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
            
            # 5. 캐시 저장
            if self.analysis_cache:
                self.analysis_cache[cache_key] = result.copy()
            
            # 6. 통계 업데이트
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
        
        # 테스트 케이스들
        test_cases = [
            ("티끌 모아 태산", 1),                    # 5글자, 단순구조 → 쉬움
            ("가는 말이 고와야 오는 말이 곱다", 1),    # 일상어, 대칭구조 → 쉬움
            ("등잔 밑이 어둡다", 2),                  # 중간길이, 은유 → 보통
            ("가자니 태산이요 돌아서자니 숭산이라", 3) # 길고복잡, 고어 → 어려움
        ]
        
        print(f"\n🔍 테스트 케이스 분석:")
        print("-" * 60)
        print(f"{'속담':<30} {'예상':<4} {'실제':<4} {'점수':<6} {'신뢰도':<8} {'설명'}")
        print("-" * 60)
        
        correct_predictions = 0
        total_tests = len(test_cases)
        
        for proverb_text, expected_level in test_cases:
            # 직접 분석 (테스트용)
            analysis_result = analyzer.calculate_final_difficulty(proverb_text)
            actual_level = analysis_result['difficulty_level']
            confidence = analysis_result['confidence']
            score = analysis_result['final_score']
            
            # 결과 출력
            proverb_short = proverb_text[:25] + "..." if len(proverb_text) > 25 else proverb_text
            status = "✅" if actual_level == expected_level else "❌"
            print(f"{proverb_short:<30} {expected_level:<4} {actual_level:<4} {score:<6.2f} {confidence:<8.1%} {status}")
            
            if actual_level == expected_level:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_tests * 100
        print(f"\n📊 테스트 결과: {correct_predictions}/{total_tests} 정확 ({accuracy:.1f}%)")
        
        # 1. 개별 속담 분석 테스트 (데이터베이스)
        print(f"\n🔍 데이터베이스 개별 속담 분석 테스트:")
        print("-" * 40)
        
        # ID로 분석
        result1 = analyzer.analyze_proverb_difficulty(proverb_id=1)
        if result1['difficulty_level'] > 0:
            print(f"✅ ID 분석 성공:")
            print(f"   속담: {result1['full_proverb']}")
            print(f"   난이도: {result1['difficulty_level']}단계 ({result1['score']}점)")
            print(f"   신뢰도: {result1['confidence']:.1%}")
            print(f"   처리시간: {result1['processing_time']:.3f}초")
            print(f"   설명: {result1['message']}")
        
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
        print(f"분석 방식: 하이브리드 (언어학적 + AI 모델)")
        
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
        python analyzer.py
    """
    print("🎯 속담 게임 - 하이브리드 난이도 분석기")
    print("🔬 분석 방식: 사용 빈도 중심 (AI 사용빈도 60% + 언어적복잡성 40%)")
    print("📊 참조 속담 방식 제거 - 임베딩 벡터 특성 직접 분석")
    print("🗄️ 데이터베이스: proverb_game.proverb (root/0000)")
    print()
    
    success = test_difficulty_analyzer()
    sys.exit(0 if success else 1)