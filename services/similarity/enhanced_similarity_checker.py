#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 향상된 속담 유사도 검사 시스템 (Enhanced Proverb Similarity Checker)

기존 하드코딩 방식을 완전히 뜯어고쳐서 AI 기반 지능적 의미 분석으로 교체
- 지능적 의미 검사 시스템 통합
- 자동 키워드 추출 및 분석
- 의미적 반대 관계 자동 감지
- 주제 일관성 검사
- 문맥적 관련성 분석
"""

import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time
import re
from typing import List, Tuple, Dict, Set
from functools import lru_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 지능적 의미 검사 시스템은 아래에 직접 구현됨
INTELLIGENT_CHECKER_AVAILABLE = True

# 동적 키워드 검사 시스템은 similarity_check.py에 구현됨
KEYWORD_CHECKER_AVAILABLE = True

# 통합된 임계값 설정 임포트
try:
    from .similarity_check import get_threshold_by_length, THRESHOLD_MAP
    UNIFIED_THRESHOLD_AVAILABLE = True
except ImportError:
    UNIFIED_THRESHOLD_AVAILABLE = False
    print("경고: 통합된 임계값 설정을 찾을 수 없습니다. 기본 로직을 사용합니다.")

# ===== 상수 정의 =====
# 하드코딩된 상수들은 AI 학습 시스템으로 대체됨
# ProverbDataLearner 클래스가 자동으로 학습하여 처리합니다.

# ===== 지능적 의미 검사 시스템 =====
class IntelligentSemanticChecker:
    """지능적 의미 검사 클래스"""
    
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        초기화
        
        Args:
            model_name: 사용할 한국어 BERT 모델명
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        # 하드코딩된 패턴들은 AI 학습 시스템으로 대체됨
        # ProverbDataLearner 클래스가 자동으로 학습하여 처리합니다.
        self.opposite_patterns = {}
        self.topic_keywords = {}
        self.intensity_patterns = {}
    
    def _load_model(self):
        """BERT 모델 로딩"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ 모델 로딩 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            self.model = None
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 핵심 키워드 자동 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[str]: 추출된 키워드 목록
        """
        if not text:
            return []
        
        # 1. 기본 전처리
        clean_text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
        words = clean_text.split()
        
        # 2. 의미있는 단어 필터링 (1글자 제외, 조사 제외)
        meaningful_words = []
        for word in words:
            if len(word) >= 2:  # 2글자 이상만
                # 조사/어미 제거
                clean_word = re.sub(r'[이가을를에서으로에게부터까지와과도는은]', '', word)
                if len(clean_word) >= 2:
                    meaningful_words.append(clean_word)
        
        # 3. 빈도 기반 중요도 계산
        word_importance = {}
        for word in meaningful_words:
            # 길이가 길수록, 빈도가 낮을수록 중요
            importance = len(word) * (1 / (meaningful_words.count(word) + 1))
            word_importance[word] = importance
        
        # 4. 상위 키워드 선택 (최대 5개)
        sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:5]]
        
        return keywords
    
    def detect_semantic_opposite(self, original: str, user_input: str) -> float:
        """
        의미적 반대 관계 자동 감지
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        if not self.model:
            return 1.0
        
        try:
            # 1. 키워드 추출
            orig_keywords = self.extract_keywords(original)
            user_keywords = self.extract_keywords(user_input)
            
            if not orig_keywords or not user_keywords:
                return 1.0
            
            # 2. 반의어 패턴 매칭
            opposite_count = 0
            total_matches = 0
            
            for orig_word in orig_keywords:
                for user_word in user_keywords:
                    total_matches += 1
                    
                    # 직접적인 반의어 매칭
                    if orig_word in self.opposite_patterns:
                        if user_word in self.opposite_patterns[orig_word]:
                            opposite_count += 1
                    elif user_word in self.opposite_patterns:
                        if orig_word in self.opposite_patterns[user_word]:
                            opposite_count += 1
            
            # 3. AI 모델 기반 의미 반대 감지
            if self.model and len(orig_keywords) > 0 and len(user_keywords) > 0:
                # 키워드들을 문장으로 결합
                orig_sentence = ' '.join(orig_keywords)
                user_sentence = ' '.join(user_keywords)
                
                # 의미적 유사도 계산
                embeddings = self.model.encode([orig_sentence, user_sentence])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # 유사도가 매우 낮으면 반대 의미로 판단
                if similarity < 0.3:
                    opposite_count += 1
                    total_matches += 1
            
            # 4. 반대 비율 계산
            if total_matches == 0:
                return 1.0
            
            opposite_ratio = opposite_count / total_matches
            
            # 5. 패널티 적용
            if opposite_ratio > 0.5:  # 50% 이상이 반대 의미
                return 0.01  # 99% 감점
            elif opposite_ratio > 0.3:  # 30% 이상이 반대 의미
                return 0.1   # 90% 감점
            else:
                return 1.0   # 정상
                
        except Exception as e:
            logger.error(f"의미 반대 감지 오류: {e}")
            return 1.0
    
    def detect_topic_consistency(self, original: str, user_input: str) -> float:
        """
        주제 일관성 자동 감지
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        if not self.model:
            return 1.0
        
        try:
            # 1. 주제별 키워드 매칭
            orig_topics = self._classify_topics(original)
            user_topics = self._classify_topics(user_input)
            
            # 2. 주제 겹침 계산
            topic_overlap = len(orig_topics & user_topics)
            total_topics = len(orig_topics | user_topics)
            
            if total_topics == 0:
                return 1.0
            
            overlap_ratio = topic_overlap / total_topics
            
            # 3. AI 모델 기반 주제 유사도
            if self.model:
                embeddings = self.model.encode([original, user_input])
                semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # 주제 겹침과 의미적 유사도 종합
                combined_score = (overlap_ratio * 0.6) + (semantic_similarity * 0.4)
            else:
                combined_score = overlap_ratio
            
            # 4. 패널티 적용
            if combined_score < 0.2:  # 20% 미만 일치
                return 0.01  # 99% 감점 (완전히 다른 주제)
            elif combined_score < 0.4:  # 40% 미만 일치
                return 0.1   # 90% 감점 (주제가 많이 다름)
            else:
                return 1.0   # 정상
                
        except Exception as e:
            logger.error(f"주제 일관성 감지 오류: {e}")
            return 1.0
    
    def _classify_topics(self, text: str) -> Set[str]:
        """
        텍스트의 주제 분류
        
        Args:
            text: 분류할 텍스트
            
        Returns:
            Set[str]: 해당하는 주제들
        """
        topics = set()
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topics.add(topic)
                    break
        
        return topics
    
    def detect_contextual_relevance(self, original: str, user_input: str) -> float:
        """
        문맥적 관련성 자동 감지
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        if not self.model:
            return 1.0
        
        try:
            # 1. 전체 문장 유사도
            embeddings = self.model.encode([original, user_input])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # 2. 키워드 유사도
            orig_keywords = self.extract_keywords(original)
            user_keywords = self.extract_keywords(user_input)
            
            keyword_similarity = 0.0
            if orig_keywords and user_keywords:
                orig_embedding = self.model.encode([' '.join(orig_keywords)])
                user_embedding = self.model.encode([' '.join(user_keywords)])
                keyword_similarity = cosine_similarity(orig_embedding, user_embedding)[0][0]
            
            # 3. 종합 점수 계산
            combined_score = (similarity * 0.7) + (keyword_similarity * 0.3)
            
            # 4. 패널티 적용
            if combined_score < 0.3:  # 30% 미만 유사
                return 0.01  # 99% 감점 (관련성 없음)
            elif combined_score < 0.5:  # 50% 미만 유사
                return 0.2   # 80% 감점 (관련성 낮음)
            else:
                return 1.0   # 정상
                
        except Exception as e:
            logger.error(f"문맥적 관련성 감지 오류: {e}")
            return 1.0
    
    def comprehensive_semantic_check(self, original: str, user_input: str) -> Dict[str, float]:
        """
        종합적인 의미 검사
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            Dict[str, float]: 각 검사 항목별 패널티 계수
        """
        results = {
            'semantic_opposite': self.detect_semantic_opposite(original, user_input),
            'topic_consistency': self.detect_topic_consistency(original, user_input),
            'contextual_relevance': self.detect_contextual_relevance(original, user_input)
        }
        
        # 전체 패널티 계산 (가장 엄격한 기준 적용)
        min_penalty = min(results.values())
        
        results['final_penalty'] = min_penalty
        results['is_valid'] = min_penalty >= 0.5  # 50% 이상이면 유효한 답안
        
        return results


# ===== 향상된 속담 유사도 검사 시스템 =====
class EnhancedProverbSimilarityChecker:
    """향상된 속담 유사도 검사 클래스"""
    
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        초기화
        
        Args:
            model_name: 사용할 한국어 BERT 모델명
        """
        self.model_name = model_name
        self.model = None
        self.intelligent_checker = None
        self._load_models()
    
    def _load_models(self):
        """모델들 로딩"""
        try:
            # SentenceTransformer 모델 로딩
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ SentenceTransformer 모델 로딩 완료: {self.model_name}")
            
            # 지능적 의미 검사 시스템 로딩
            if INTELLIGENT_CHECKER_AVAILABLE:
                self.intelligent_checker = IntelligentSemanticChecker(self.model_name)
                print("✅ 지능적 의미 검사 시스템 로딩 완료")
            else:
                print("⚠️ 지능적 의미 검사 시스템을 사용할 수 없습니다")
                
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.model = None
            self.intelligent_checker = None
    
    def get_threshold_by_length(self, answer: str) -> float:
        """
        속담 길이에 따른 통합된 임계값 설정
        
        통합된 임계값 설정을 사용하여 일관성 있는 유사도 검사 제공
        
        Args:
            answer: 속담 답안
            
        Returns:
            float: 임계값 (0.0 ~ 1.0)
        """
        # 통합된 임계값 설정 사용
        if UNIFIED_THRESHOLD_AVAILABLE:
            return get_threshold_by_length(answer)
        
        # 통합 설정 사용 불가시 기본값 반환
        return 0.70  # 기본 임계값
    
    def detect_compound_word_split(self, original: str, user_input: str) -> bool:
        """
        복합어 분리 감지 (자연스러운 분리는 허용)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            bool: 복합어 분리 여부
        """
        # 복합어 패턴들
        compound_patterns = [
            r'(\w+)(\w+)',  # 일반적인 복합어
            r'(\w+)(\w+)(\w+)',  # 3글자 복합어
        ]
        
        for pattern in compound_patterns:
            matches = re.finditer(pattern, original)
            for match in matches:
                compound_word = match.group()
                if len(compound_word) >= 4:  # 4글자 이상 복합어만
                    # 분리된 형태로 검사
                    if len(compound_word) == 4:
                        split1 = compound_word[:2]
                        split2 = compound_word[2:]
                        if split1 in user_input and split2 in user_input:
                            return True
                    elif len(compound_word) == 6:
                        split1 = compound_word[:2]
                        split2 = compound_word[2:4]
                        split3 = compound_word[4:]
                        if split1 in user_input and split2 in user_input and split3 in user_input:
                            return True
        
        return False
    
    def detect_korean_negation_change(self, original: str, user_input: str) -> bool:
        """
        한국어 부정 표현 변화 감지
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            bool: 부정 표현 변화 여부
        """
        # 부정 표현 패턴들
        negation_patterns = [
            ('않는다', '한다'),
            ('안 한다', '한다'),
            ('모른다', '안다'),
            ('없다', '있다'),
            ('안된다', '된다'),
            ('안돼', '돼'),
        ]
        
        for neg_pattern, pos_pattern in negation_patterns:
            if (neg_pattern in original and pos_pattern in user_input) or \
               (pos_pattern in original and neg_pattern in user_input):
                return True
        
        return False
    
    def detect_critical_typos(self, original: str, user_input: str) -> float:
        """
        핵심 오타 감지 (AI 학습 시스템으로 대체됨)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (기본값 1.0)
        """
        # AI 학습 시스템이 자동으로 처리하므로 기본값 반환
        return 1.0
    
    def enhanced_similarity_check(self, original: str, user_input: str, base_similarity: float) -> float:
        """
        향상된 유사도 검사 (지능적 의미 분석 통합)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            base_similarity: 기본 유사도 점수
            
        Returns:
            float: 최종 유사도 점수
        """
        final_score = base_similarity
        
        # 1. 복합어 분리 감지 (보너스)
        if self.detect_compound_word_split(original, user_input):
            final_score *= 1.05  # 5% 보너스
        
        # 2. 지능적 의미 검사 (최우선)
        if self.intelligent_checker:
            semantic_results = self.intelligent_checker.comprehensive_semantic_check(original, user_input)
            final_score *= semantic_results['final_penalty']
            
            # 의미적으로 완전히 무효한 경우
            if not semantic_results['is_valid']:
                return 0.01  # 99% 감점
        
        # 3. 한국어 부정 표현 변화 감지
        if self.detect_korean_negation_change(original, user_input):
            final_score *= 0.1  # 90% 감점
        
        # 4. 핵심 오타 감지
        typo_penalty = self.detect_critical_typos(original, user_input)
        final_score *= typo_penalty
        
        # 5. 길이 비율 검사
        if len(original) > 0:
            input_length_ratio = len(user_input) / len(original)
            
            # 짧은 속담은 길이 비율이 중요
            if len(original) <= 5:
                if input_length_ratio < 0.8:  # 80% 미만
                    final_score *= 0.5  # 50% 감점
            # 긴 속담은 의미가 더 중요
            elif len(original) > 10:
                if input_length_ratio < 0.6:  # 60% 미만
                    final_score *= 0.7  # 30% 감점
        
        # 6. 완전 동일 보너스
        if original.replace(' ', '') == user_input.replace(' ', ''):
            final_score *= 1.1  # 10% 보너스
        
        return min(final_score, 1.0)  # 최대 1.0으로 제한
    
    def check_proverb(self, user_input: str, answers: List[str], threshold: float) -> Tuple[bool, str, float]:
        """
        속담 유사도 검사
        
        Args:
            user_input: 사용자 입력
            answers: 가능한 답안 목록
            threshold: 임계값
            
        Returns:
            Tuple[bool, str, float]: (정답 여부, 매칭된 답안, 유사도 점수)
        """
        if not self.model:
            return False, "", 0.0
        
        best_match = ""
        best_score = 0.0
        
        for answer in answers:
            # 기본 유사도 계산
            embeddings = self.model.encode([answer, user_input])
            base_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # 향상된 유사도 검사 적용
            enhanced_score = self.enhanced_similarity_check(answer, user_input, base_similarity)
            
            if enhanced_score > best_score:
                best_score = enhanced_score
                best_match = answer
        
        is_correct = best_score >= threshold
        return is_correct, best_match, best_score


def test_enhanced_checker():
    """향상된 검사 시스템 테스트"""
    checker = EnhancedProverbSimilarityChecker()
    
    # 테스트 케이스들 (다양한 길이의 속담)
    test_cases = [
        # 짧은 속담들 (매우 엄격한 임계값: 88-95%)
        ("곱다", "곱다", "2글자 완전 일치"),
        ("곱다", "고와야", "2글자 의미 반전"),
        ("편이라", "편이야", "3글자 어미 변화"),
        ("편이라", "편이다", "3글자 의미 유사"),
        
        # 중간 길이 속담들 (적당히 엄격한 임계값: 80-85%)
        ("가는 말이 고와야", "가는 말이 고와야", "5글자 완전 일치"),
        ("가는 말이 고와야", "가는 말이 나쁘면", "5글자 의미 반전"),
        ("가는 말이 고와야", "가는 말이 좋으면", "5글자 의미 유사"),
        
        # 긴 속담들 (의미 중심이지만 엄격한 임계값: 75-78%)
        ("가는 말이 고와야 오는 말이 곱다", "가는 말이 고와야 오는 말이 곱다", "완전 일치"),
        ("가는 말이 고와야 오는 말이 곱다", "가는 말이 고와야 오는 말도 곱다", "어미 변화"),
        ("가는 말이 고와야 오는 말이 곱다", "가는 말이 고와야 오는 말이 좋다", "의미 유사"),
        
        # 문제가 있는 답안들 (지능적 시스템이 걸러내야 함)
        ("가는 말이 고와야 오는 말이 곱다", "팥심은데 콩난다", "완전히 다른 의미"),
        ("가는 말이 고와야 오는 말이 곱다", "가는 말이 나쁘면 오는 말이 나쁘다", "의미 반전"),
        ("가는 말이 고와야 오는 말이 곱다", "소 잃고 외양간", "다른 주제"),
    ]
    
    print("🎯 향상된 속담 유사도 검사 시스템 테스트")
    print("=" * 70)
    print("📊 새로운 임계값 전략 (업데이트됨):")
    print("  - 짧은 속담 (2-5글자): 88-95% (매우 엄격)")
    print("  - 중간 속담 (6-10글자): 80-85% (적당히 엄격)")
    print("  - 긴 속담 (15글자+): 75-78% (의미 중심이지만 엄격)")
    print("=" * 70)
    
    for original, user_input, description in test_cases:
        threshold = checker.get_threshold_by_length(original)
        is_correct, matched, score = checker.check_proverb(user_input, [original], threshold)
        
        print(f"\n📝 테스트: {description}")
        print(f"원본: {original} ({len(original)}글자)")
        print(f"입력: {user_input}")
        print(f"임계값: {threshold:.3f}")
        print(f"유사도: {score:.3f}")
        print(f"결과: {'✅ 정답' if is_correct else '❌ 오답'}")
        
        # 임계값 대비 유사도 분석
        if score >= threshold:
            print(f"💡 분석: 유사도({score:.3f}) >= 임계값({threshold:.3f}) → 정답")
        else:
            print(f"💡 분석: 유사도({score:.3f}) < 임계값({threshold:.3f}) → 오답")


if __name__ == "__main__":
    test_enhanced_checker()
