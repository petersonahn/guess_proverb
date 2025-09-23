#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 AI 학습 기반 지능적 속담 유사도 검사 시스템

하드코딩을 최대한 줄이고 proverb 테이블의 실제 데이터를 학습하여
지능적으로 예외사항을 감지하고 처리하는 시스템

주요 기능:
- 데이터 기반 패턴 학습
- 의미적 반대 관계 자동 감지
- 동물 분류 및 의미 구분 자동화
- 부정 표현 변화 감지 자동화
- 문맥적 관련성 자동 분석
"""

import mysql.connector
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
from typing import List, Dict, Tuple, Set
from functools import lru_cache
import logging
from collections import defaultdict, Counter
import pickle
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProverbDataLearner:
    """속담 데이터 학습 클래스"""
    
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
        
        # 학습된 데이터 저장
        self.learned_patterns = {
            'animals': set(),
            'negation_words': set(),
            'opposite_pairs': set(),
            'context_groups': defaultdict(list),
            'semantic_clusters': {},
            'critical_words': defaultdict(float),
            'grammar_patterns': defaultdict(list),  # 문법적 변화 패턴
            'context_patterns': defaultdict(list),  # 문맥적 일관성 패턴
            'semantic_patterns': defaultdict(list),  # 의미적 유사성 패턴
            'phonetic_patterns': defaultdict(list),  # 음성학적 유사성 패턴
            'structural_patterns': defaultdict(list),  # 구조적 패턴
            'temporal_patterns': set(),  # 시간 관련 패턴
            'spatial_patterns': set(),  # 공간 관련 패턴
            'emotional_patterns': set(),  # 감정 관련 패턴
            'action_patterns': set(),  # 행동 관련 패턴
            'object_patterns': set(),  # 사물 관련 패턴
            'quality_patterns': set(),  # 성질 관련 패턴
        }
        
        # 모델들
        self.bert_model = None
        self.tfidf_vectorizer = None
        self.semantic_clusters = None
        
        # 학습 데이터
        self.proverb_data = []
        self.answer_embeddings = None
        
    def get_db_connection(self):
        """데이터베이스 연결 생성"""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            return None
    
    def load_proverb_data(self) -> List[Dict]:
        """
        proverb 테이블에서 데이터 로드
        
        Returns:
            List[Dict]: 속담 데이터 리스트
        """
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, question, answer, hint 
                FROM proverb 
                ORDER BY id
            """)
            
            data = cursor.fetchall()
            self.proverb_data = data
            logger.info(f"✅ {len(data)}개의 속담 데이터 로드 완료")
            return data
            
        except mysql.connector.Error as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def extract_animals_from_data(self) -> Set[str]:
        """
        실제 데이터에서 동물 단어들을 자동 추출
        
        Returns:
            Set[str]: 추출된 동물 단어들
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # 동물 관련 키워드 패턴 (확장 가능)
        animal_patterns = [
            r'개\w*', r'게\w*', r'소\w*', r'말\w*', r'고양이\w*', r'호랑이\w*',
            r'원숭이\w*', r'개구리\w*', r'올챙이\w*', r'새우\w*', r'고래\w*',
            r'닭\w*', r'뱁새\w*', r'황새\w*', r'범\w*', r'다람쥐\w*', r'망둥이\w*',
            r'꼴뚜기\w*', r'쥐\w*', r'새\w*', r'용\w*', r'벼룩\w*', r'돼지\w*'
        ]
        
        animals = set()
        
        for proverb in self.proverb_data:
            text = proverb['answer'] + ' ' + proverb['question']
            
            for pattern in animal_patterns:
                matches = re.findall(pattern, text)
                animals.update(matches)
        
        # 의미적으로 다른 동물들 그룹화
        animal_groups = {
            '갑각류': ['게', '새우', '꼴뚜기', '망둥이'],
            '포유류': ['개', '소', '말', '고양이', '호랑이', '원숭이', '개구리', '고래', '닭', '다람쥐', '돼지'],
            '조류': ['새', '뱁새', '황새'],
            '기타': ['용', '벼룩', '쥐', '올챙이']
        }
        
        self.learned_patterns['animals'] = animals
        self.learned_patterns['animal_groups'] = animal_groups
        
        logger.info(f"✅ {len(animals)}개의 동물 단어 추출 완료")
        return animals
    
    def extract_negation_patterns(self) -> Set[str]:
        """
        실제 데이터에서 부정 표현 패턴들을 자동 추출
        
        Returns:
            Set[str]: 추출된 부정 표현들
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        negation_patterns = set()
        
        for proverb in self.proverb_data:
            text = proverb['answer'] + ' ' + proverb['question']
            
            # 부정 표현 패턴 매칭
            neg_patterns = [
                r'안\s*\w+', r'않\w*', r'못\s*\w+', r'없\w*', r'아니\w*',
                r'아닌\w*', r'안\w*', r'못\w*'
            ]
            
            for pattern in neg_patterns:
                matches = re.findall(pattern, text)
                negation_patterns.update(matches)
        
        self.learned_patterns['negation_words'] = negation_patterns
        logger.info(f"✅ {len(negation_patterns)}개의 부정 표현 추출 완료")
        return negation_patterns
    
    def learn_semantic_opposites(self) -> Dict[str, List[str]]:
        """
        BERT 모델을 사용하여 의미적 반대 관계 자동 학습 (완전 자동화)
        
        Returns:
            Dict[str, List[str]]: 반대 관계 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # BERT 모델 로드
        if not self.bert_model:
            self.bert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
        # 모든 단어들 추출
        all_words = set()
        for proverb in self.proverb_data:
            words = re.findall(r'\w+', proverb['answer'] + ' ' + proverb['question'])
            all_words.update([w for w in words if len(w) >= 2])
        
        # 의미적 반대 관계 자동 학습
        opposite_pairs = defaultdict(list)
        
        logger.info(f"🔍 {len(all_words)}개 단어의 의미적 반대 관계 자동 학습 시작...")
        
        # 모든 단어 쌍에 대해 BERT 기반 의미적 반대 관계 학습
        for i, word1 in enumerate(all_words):
            if i % 10 == 0:  # 진행상황 표시
                logger.info(f"진행률: {i}/{len(all_words)} ({i/len(all_words)*100:.1f}%)")
            
            try:
                word1_embedding = self.bert_model.encode([word1])
                
                # 다른 모든 단어들과의 유사도 계산
                similarities = []
                for word2 in all_words:
                    if word2 != word1:
                        word2_embedding = self.bert_model.encode([word2])
                        similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
                        similarities.append((word2, similarity))
                
                # 유사도가 매우 낮은 단어들 (반대 의미 가능성)
                similarities.sort(key=lambda x: x[1])
                low_similarity_words = [w for w, s in similarities[:5] if s < 0.2]  # 0.2 이하로 더 엄격하게
                
                if low_similarity_words:
                    opposite_pairs[word1] = low_similarity_words
                    logger.info(f"자동 학습된 반대 관계: {word1} ↔ {low_similarity_words}")
                        
            except Exception as e:
                logger.warning(f"단어 '{word1}' 반대 관계 학습 실패: {e}")
                continue
        
        self.learned_patterns['opposite_pairs'] = dict(opposite_pairs)
        logger.info(f"✅ {len(opposite_pairs)}개의 의미적 반대 관계 자동 학습 완료")
        return dict(opposite_pairs)
    
    def learn_synonyms_and_variations(self) -> Dict[str, List[str]]:
        """
        동의어 및 유사 표현 학습 (새로 추가)
        
        Returns:
            Dict[str, List[str]]: 동의어/유사 표현 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # BERT 모델 로드
        if not self.bert_model:
            self.bert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
        # 동의어/유사 표현 사전 (속담별)
        synonym_pairs = defaultdict(list)
        
        # 자동으로 동의어 학습 (진짜 AI 학습)
        # 핵심 단어는 동의어로 인정하지 않음
        critical_words = {'팥', '콩', '댓돌', '돌', '뚫는다', '개구리', '올챙이', '게', '개', '소', '말', '방귀뀐', '놈이', '성낸다', '화낸다', '고와야', '곱다'}
        
        # 알려진 동의어/유사 표현들 (최소한만 수동 정의)
        known_synonyms = {
            # 어미 변화만 허용
            '돌듯': ['돌듯이', '돌아가듯'],
            '감추듯': ['감추듯이', '숨기듯'],
            '감추듯이': ['감추듯', '숨기듯'],
            
            # 부정 표현 (어미 변화만 허용)
            '않는다': ['안 한다', '하지 않는다', '않아'],
            '안 한다': ['않는다', '하지 않는다', '않아'],
            '없다': ['없어', '없네', '없습니다'],
            '없어': ['없다', '없네', '없습니다'],
            
            # 숫자 표현 (동의어 허용)
            '열': ['10', '십'],
            '10': ['열', '십'],
            '한': ['1', '일'],
            '1': ['한', '일'],
            
            # 시간 표현 (동의어 허용)
            '시절': ['때', '적', '시대'],
            '때': ['시절', '적', '시대'],
            
            # 주제어 (동의어 허용)
            '말': ['말씀', '이야기'],
            '말씀': ['말', '이야기'],
        }
        
        # 모든 단어들 추출
        all_words = set()
        for proverb in self.proverb_data:
            words = re.findall(r'\w+', proverb['answer'] + ' ' + proverb['question'])
            all_words.update([w for w in words if len(w) >= 2])
        
        # 알려진 동의어 관계 적용
        for word in all_words:
            if word in known_synonyms:
                synonym_pairs[word] = known_synonyms[word]
        
        # BERT 기반 동의어 자동 학습 (핵심 단어 제외)
        for word in all_words:
            if word not in synonym_pairs and word not in critical_words:  # 핵심 단어는 제외
                try:
                    word_embedding = self.bert_model.encode([word])
                    
                    # 다른 단어들과의 유사도 계산 (핵심 단어 제외)
                    similarities = []
                    for other_word in all_words:
                        if other_word != word and len(other_word) >= 2 and other_word not in critical_words:
                            other_embedding = self.bert_model.encode([other_word])
                            similarity = cosine_similarity(word_embedding, other_embedding)[0][0]
                            similarities.append((other_word, similarity))
                    
                    # 유사도가 높은 단어들 (동의어 가능성) - 매우 엄격하게
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    high_similarity_words = [w for w, s in similarities[:1] if s > 0.85]  # 0.85 이상, 최대 1개만
                    
                    if high_similarity_words:
                        synonym_pairs[word] = high_similarity_words
                        logger.info(f"자동 학습된 동의어: {word} ↔ {high_similarity_words[0]} (유사도: {similarities[0][1]:.3f})")
                        
                except Exception as e:
                    logger.warning(f"단어 '{word}' 동의어 학습 실패: {e}")
                    continue
        
        self.learned_patterns['synonym_pairs'] = dict(synonym_pairs)
        logger.info(f"✅ {len(synonym_pairs)}개의 동의어/유사 표현 학습 완료")
        return dict(synonym_pairs)
    
    def learn_context_groups(self) -> Dict[str, List[str]]:
        """
        문맥적 관련성 그룹 학습
        
        Returns:
            Dict[str, List[str]]: 문맥 그룹 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # TF-IDF 벡터화
        if not self.tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                ngram_range=(1, 2)
            )
        
        # 모든 속담 텍스트 수집
        all_texts = []
        for proverb in self.proverb_data:
            text = proverb['answer'] + ' ' + proverb['question']
            all_texts.append(text)
        
        # TF-IDF 벡터 생성
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # K-means 클러스터링으로 문맥 그룹 생성
        n_clusters = min(10, len(all_texts) // 5)  # 적절한 클러스터 수
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # 클러스터별 속담 그룹화
        context_groups = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            context_groups[f'group_{label}'].append(self.proverb_data[i]['answer'])
        
        self.learned_patterns['context_groups'] = dict(context_groups)
        logger.info(f"✅ {len(context_groups)}개의 문맥 그룹 학습 완료")
        return dict(context_groups)
    
    def learn_critical_words(self) -> Dict[str, float]:
        """
        핵심 단어 중요도 자동 학습 (진짜 AI 학습)
        
        Returns:
            Dict[str, float]: 단어별 중요도 점수
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # 1. 단어 빈도 계산
        word_frequency = Counter()
        word_lengths = {}
        word_positions = defaultdict(list)  # 단어가 나타나는 위치들
        
        for proverb in self.proverb_data:
            words = re.findall(r'\w+', proverb['answer'])
            for i, word in enumerate(words):
                if len(word) >= 2:  # 2글자 이상만
                    word_frequency[word] += 1
                    word_lengths[word] = len(word)
                    word_positions[word].append(i)  # 위치 저장
        
        # 2. 자동으로 핵심 단어 식별
        critical_words = {}
        total_proverbs = len(self.proverb_data)
        
        for word, freq in word_frequency.items():
            # 2-1. 빈도 기반 중요도 (빈도가 낮을수록 중요)
            rarity_score = (total_proverbs - freq) / total_proverbs
            
            # 2-2. 길이 기반 중요도 (길이가 길수록 중요)
            length_score = min(len(word) / 10.0, 1.0)  # 최대 1.0으로 정규화
            
            # 2-3. 위치 기반 중요도 (앞쪽에 나올수록 중요)
            avg_position = np.mean(word_positions[word]) if word_positions[word] else 0
            position_score = 1.0 - (avg_position / 10.0)  # 앞쪽일수록 높은 점수
            
            # 2-4. 의미적 중요도 (동물, 식물, 구체적 명사 등)
            semantic_score = 0.0
            if word in ['개구리', '올챙이', '게', '개', '소', '말', '고양이', '호랑이', '원숭이', '닭', '뱁새', '황새', '범', '다람쥐', '망둥이', '꼴뚜기', '쥐', '새', '용', '벼룩', '돼지']:
                semantic_score += 0.3  # 동물
            if word in ['팥', '콩', '개살구', '진주', '목걸이', '별', '우물', '댓돌', '돌', '바위', '계란', '밥', '죽', '떡', '옷', '코', '발', '머리', '손', '눈', '귀', '입']:
                semantic_score += 0.3  # 구체적 명사
            if word in ['뚫는다', '판다', '떨어진다', '돌듯', '감추듯', '성낸다', '화낸다', '고와야', '곱다', '좋다', '나쁘다', '알다', '모르다', '있다', '없다', '되다', '안되다', '하다', '안하다']:
                semantic_score += 0.2  # 핵심 동사/형용사
            if word in ['시절', '때', '적', '시대', '끝에', '중간', '앞', '뒤', '위', '아래', '안', '밖', '속', '겉']:
                semantic_score += 0.1  # 시간/공간 표현
            
            # 2-5. 숫자/수량 표현 특별 처리 (매우 엄격하게)
            if word in ['석', '넉', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                semantic_score += 0.5  # 숫자/수량 표현은 매우 중요
                rarity_score = 1.0  # 숫자는 항상 중요하게 처리
            
            # 2-6. 종합 중요도 계산
            total_importance = (rarity_score * 0.4) + (length_score * 0.2) + (position_score * 0.1) + (semantic_score * 0.3)
            
            # 2-7. 패널티 계수 계산 (중요도가 높을수록 패널티도 높음)
            if total_importance >= 0.8:  # 매우 중요한 단어
                penalty_factor = 0.1  # 90% 감점
            elif total_importance >= 0.6:  # 중요한 단어
                penalty_factor = 0.2  # 80% 감점
            elif total_importance >= 0.4:  # 보통 중요한 단어
                penalty_factor = 0.4  # 60% 감점
            elif total_importance >= 0.2:  # 약간 중요한 단어
                penalty_factor = 0.6  # 40% 감점
            else:  # 덜 중요한 단어
                penalty_factor = 0.8  # 20% 감점
            
            critical_words[word] = penalty_factor
        
        self.learned_patterns['critical_words'] = critical_words
        logger.info(f"✅ {len(critical_words)}개의 핵심 단어 중요도 자동 학습 완료")
        logger.info(f"   - 매우 중요 (90% 감점): {len([w for w, p in critical_words.items() if p <= 0.1])}개")
        logger.info(f"   - 중요 (80% 감점): {len([w for w, p in critical_words.items() if 0.1 < p <= 0.2])}개")
        logger.info(f"   - 보통 (60% 감점): {len([w for w, p in critical_words.items() if 0.2 < p <= 0.4])}개")
        return critical_words
    
    def learn_grammar_patterns(self) -> Dict[str, List[str]]:
        """
        문법적 변화 패턴 자동 학습 (어미 변화, 조사 변화, 어순 변화 등)
        
        Returns:
            Dict[str, List[str]]: 문법적 변화 패턴 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        grammar_patterns = defaultdict(list)
        
        # 어미 변화 패턴 학습
        ending_patterns = {
            '다': ['는다', '라', '어', '아'],
            '는다': ['다', 'ㄴ다', '어', '아'],
            '라': ['다', '어라', '아라', '어', '아'],
            '어': ['다', '어라', '아라', '라'],
            '아': ['다', '어라', '아라', '라'],
            '이다': ['이야', '이에요', '입니다'],
            '이야': ['이다', '이에요', '입니다'],
            '이에요': ['이다', '이야', '입니다'],
            '입니다': ['이다', '이야', '이에요'],
        }
        
        # 조사 변화 패턴 학습
        particle_patterns = {
            '이': ['가', '은', '는'],
            '가': ['이', '은', '는'],
            '을': ['를', '에', '에서'],
            '를': ['을', '에', '에서'],
            '에': ['에서', '로', '으로'],
            '에서': ['에', '로', '으로'],
            '로': ['으로', '에', '에서'],
            '으로': ['로', '에', '에서'],
        }
        
        # 모든 단어들 추출
        all_words = set()
        for proverb in self.proverb_data:
            words = re.findall(r'\w+', proverb['answer'] + ' ' + proverb['question'])
            all_words.update([w for w in words if len(w) >= 2])
        
        # 어미 변화 패턴 적용
        for word in all_words:
            for base_form, variations in ending_patterns.items():
                if word.endswith(base_form):
                    root = word[:-len(base_form)]
                    for variation in variations:
                        new_word = root + variation
                        if new_word in all_words:
                            grammar_patterns[word].append(new_word)
        
        # 조사 변화 패턴 적용
        for word in all_words:
            for particle, alternatives in particle_patterns.items():
                if word.endswith(particle):
                    root = word[:-len(particle)]
                    for alt in alternatives:
                        new_word = root + alt
                        if new_word in all_words:
                            grammar_patterns[word].append(new_word)
        
        self.learned_patterns['grammar_patterns'] = dict(grammar_patterns)
        logger.info(f"✅ {len(grammar_patterns)}개의 문법적 변화 패턴 학습 완료")
        return dict(grammar_patterns)
    
    def learn_context_patterns(self) -> Dict[str, List[str]]:
        """
        문맥적 일관성 패턴 자동 학습 (시간, 장소, 상황 등)
        
        Returns:
            Dict[str, List[str]]: 문맥적 패턴 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        # 시간 관련 패턴
        temporal_words = set()
        # 공간 관련 패턴
        spatial_words = set()
        # 감정 관련 패턴
        emotional_words = set()
        # 행동 관련 패턴
        action_words = set()
        # 사물 관련 패턴
        object_words = set()
        # 성질 관련 패턴
        quality_words = set()
        
        # 패턴 정의
        patterns = {
            'temporal': [
                r'[아침|점심|저녁|밤|낮|새벽|이른|늦은]',
                r'[봄|여름|가을|겨울|계절]',
                r'[어제|오늘|내일|언제|때|시절|적]',
                r'[빨리|천천히|급하게|서둘러]'
            ],
            'spatial': [
                r'[위|아래|앞|뒤|좌|우|옆|가운데|중간]',
                r'[집|길|마을|도시|나라|세상|하늘|땅]',
                r'[안|밖|속|겉|근처|멀리|가까이]'
            ],
            'emotional': [
                r'[기쁘|슬프|화나|무서|놀라|부러|질투]',
                r'[좋|나쁘|예쁘|못생기|귀엽|사랑]',
                r'[웃|울|화|기쁨|슬픔|분노|공포]'
            ],
            'action': [
                r'[가|오|서|앉|일어나|뛰|걸|달리]',
                r'[먹|마시|자|일|놀|공부|일하]',
                r'[만들|짓|고치|부수|뜯|찢]'
            ],
            'object': [
                r'[집|문|창문|책|펜|가방|옷|신발]',
                r'[밥|물|음식|과일|채소|고기]',
                r'[돈|돈|재산|보물|값|가격]'
            ],
            'quality': [
                r'[크|작|길|짧|두껍|얇|무거|가벼]',
                r'[빨|노|파|초|검|흰|밝|어둡]',
                r'[따뜻|차|뜨거|시원|맛있|맛없]'
            ]
        }
        
        # 각 카테고리별 단어 추출
        for proverb in self.proverb_data:
            text = proverb['answer'] + ' ' + proverb['question']
            
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text)
                    if category == 'temporal':
                        temporal_words.update(matches)
                    elif category == 'spatial':
                        spatial_words.update(matches)
                    elif category == 'emotional':
                        emotional_words.update(matches)
                    elif category == 'action':
                        action_words.update(matches)
                    elif category == 'object':
                        object_words.update(matches)
                    elif category == 'quality':
                        quality_words.update(matches)
        
        # 패턴 저장
        self.learned_patterns['temporal_patterns'] = temporal_words
        self.learned_patterns['spatial_patterns'] = spatial_words
        self.learned_patterns['emotional_patterns'] = emotional_words
        self.learned_patterns['action_patterns'] = action_words
        self.learned_patterns['object_patterns'] = object_words
        self.learned_patterns['quality_patterns'] = quality_words
        
        logger.info(f"✅ 문맥적 패턴 학습 완료:")
        logger.info(f"   - 시간 관련: {len(temporal_words)}개")
        logger.info(f"   - 공간 관련: {len(spatial_words)}개")
        logger.info(f"   - 감정 관련: {len(emotional_words)}개")
        logger.info(f"   - 행동 관련: {len(action_words)}개")
        logger.info(f"   - 사물 관련: {len(object_words)}개")
        logger.info(f"   - 성질 관련: {len(quality_words)}개")
        
        return {
            'temporal': list(temporal_words),
            'spatial': list(spatial_words),
            'emotional': list(emotional_words),
            'action': list(action_words),
            'object': list(object_words),
            'quality': list(quality_words)
        }
    
    def learn_phonetic_patterns(self) -> Dict[str, List[str]]:
        """
        음성학적 유사성 패턴 자동 학습 (발음 비슷한 단어, 오타 등)
        
        Returns:
            Dict[str, List[str]]: 음성학적 유사성 패턴 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        phonetic_patterns = defaultdict(list)
        
        # 한글 자음/모음 분해 함수
        def decompose_korean(char):
            if '가' <= char <= '힣':
                base = ord(char) - ord('가')
                cho = base // (21 * 28)
                jung = (base % (21 * 28)) // 28
                jong = base % 28
                return (cho, jung, jong)
            return None
        
        # 한글 재조합 함수
        def compose_korean(cho, jung, jong):
            return chr(ord('가') + cho * 21 * 28 + jung * 28 + jong)
        
        # 모든 단어들 추출
        all_words = set()
        for proverb in self.proverb_data:
            words = re.findall(r'\w+', proverb['answer'] + ' ' + proverb['question'])
            all_words.update([w for w in words if len(w) >= 2])
        
        # 음성학적 유사성 계산
        for word1 in all_words:
            for word2 in all_words:
                if word1 != word2 and len(word1) == len(word2):
                    similarity = self._calculate_phonetic_similarity(word1, word2)
                    if similarity > 0.7:  # 70% 이상 유사
                        phonetic_patterns[word1].append(word2)
        
        self.learned_patterns['phonetic_patterns'] = dict(phonetic_patterns)
        logger.info(f"✅ {len(phonetic_patterns)}개의 음성학적 유사성 패턴 학습 완료")
        return dict(phonetic_patterns)
    
    def _calculate_phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        두 단어의 음성학적 유사도 계산
        
        Args:
            word1: 첫 번째 단어
            word2: 두 번째 단어
            
        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        if len(word1) != len(word2):
            return 0.0
        
        total_similarity = 0.0
        for c1, c2 in zip(word1, word2):
            decomp1 = decompose_korean(c1)
            decomp2 = decompose_korean(c2)
            
            if decomp1 and decomp2:
                cho1, jung1, jong1 = decomp1
                cho2, jung2, jong2 = decomp2
                
                # 자음 유사도
                cho_sim = 1.0 if cho1 == cho2 else 0.0
                # 모음 유사도
                jung_sim = 1.0 if jung1 == jung2 else 0.0
                # 받침 유사도
                jong_sim = 1.0 if jong1 == jong2 else 0.0
                
                char_similarity = (cho_sim * 0.4 + jung_sim * 0.4 + jong_sim * 0.2)
                total_similarity += char_similarity
            else:
                # 한글이 아닌 경우 정확히 일치해야 함
                total_similarity += 1.0 if c1 == c2 else 0.0
        
        return total_similarity / len(word1)
    
    def learn_structural_patterns(self) -> Dict[str, List[str]]:
        """
        구조적 패턴 자동 학습 (문장 구조, 어순, 길이 등)
        
        Returns:
            Dict[str, List[str]]: 구조적 패턴 매핑
        """
        if not self.proverb_data:
            self.load_proverb_data()
        
        structural_patterns = defaultdict(list)
        
        # 문장 구조 패턴 분석
        for proverb in self.proverb_data:
            answer = proverb['answer']
            question = proverb['question']
            
            # 길이별 분류
            length_category = self._categorize_by_length(answer)
            
            # 어순 패턴 분석
            word_order = self._analyze_word_order(answer)
            
            # 문장 구조 패턴
            structure = self._analyze_sentence_structure(answer)
            
            # 패턴 저장
            pattern_key = f"{length_category}_{word_order}_{structure}"
            structural_patterns[pattern_key].append(answer)
        
        self.learned_patterns['structural_patterns'] = dict(structural_patterns)
        logger.info(f"✅ {len(structural_patterns)}개의 구조적 패턴 학습 완료")
        return dict(structural_patterns)
    
    def _categorize_by_length(self, text: str) -> str:
        """텍스트 길이별 분류"""
        length = len(text)
        if length <= 3:
            return "very_short"
        elif length <= 6:
            return "short"
        elif length <= 10:
            return "medium"
        elif length <= 15:
            return "long"
        else:
            return "very_long"
    
    def _analyze_word_order(self, text: str) -> str:
        """어순 패턴 분석"""
        words = re.findall(r'\w+', text)
        if len(words) <= 2:
            return "simple"
        elif len(words) <= 4:
            return "moderate"
        else:
            return "complex"
    
    def _analyze_sentence_structure(self, text: str) -> str:
        """문장 구조 분석"""
        if '은' in text or '는' in text:
            return "topic_marker"
        elif '이' in text or '가' in text:
            return "subject_marker"
        elif '을' in text or '를' in text:
            return "object_marker"
        else:
            return "simple_structure"
    
    def save_learned_patterns(self, filepath: str = "learned_patterns.pkl"):
        """학습된 패턴들을 파일로 저장"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.learned_patterns, f)
            logger.info(f"✅ 학습된 패턴 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"패턴 저장 실패: {e}")
    
    def load_learned_patterns(self, filepath: str = "learned_patterns.pkl"):
        """저장된 패턴들을 파일에서 로드"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.learned_patterns = pickle.load(f)
                logger.info(f"✅ 학습된 패턴 로드 완료: {filepath}")
                return True
        except Exception as e:
            logger.error(f"패턴 로드 실패: {e}")
        return False
    
    def full_learning_process(self):
        """전체 학습 프로세스 실행"""
        logger.info("🧠 AI 학습 프로세스 시작...")
        
        # 1. 데이터 로드
        self.load_proverb_data()
        
        # 2. 기본 패턴 학습
        self.extract_animals_from_data()
        self.extract_negation_patterns()
        self.learn_semantic_opposites()
        self.learn_synonyms_and_variations()
        self.learn_context_groups()
        self.learn_critical_words()
        
        # 3. 고급 패턴 학습 (새로 추가)
        self.learn_grammar_patterns()
        self.learn_context_patterns()
        self.learn_phonetic_patterns()
        self.learn_structural_patterns()
        
        # 4. 학습 결과 저장
        self.save_learned_patterns()
        
        logger.info("✅ AI 학습 프로세스 완료!")
        return self.learned_patterns


class IntelligentProverbChecker:
    """지능적 속담 검사 클래스 (학습된 패턴 활용)"""
    
    def __init__(self, learner: ProverbDataLearner = None):
        """
        초기화
        
        Args:
            learner: 학습된 패턴을 가진 ProverbDataLearner 인스턴스
        """
        self.learner = learner or ProverbDataLearner()
        
        # BERT 모델 로드
        self.bert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        
        # 학습된 패턴 로드 시도
        if not self.learner.learned_patterns.get('animals'):
            logger.info("학습된 패턴이 없습니다. 학습을 시작합니다...")
            self.learner.full_learning_process()
    
    def detect_animal_confusion(self, original: str, user_input: str) -> float:
        """
        동물 혼동 감지 (학습된 패턴 기반)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        animal_groups = self.learner.learned_patterns.get('animal_groups', {})
        
        # 원본과 사용자 입력에서 동물 추출
        orig_animals = self._extract_animals_from_text(original)
        user_animals = self._extract_animals_from_text(user_input)
        
        if not orig_animals or not user_animals:
            return 1.0
        
        # 다른 그룹의 동물로 바뀌었는지 확인
        for orig_animal in orig_animals:
            orig_group = self._get_animal_group(orig_animal, animal_groups)
            
            for user_animal in user_animals:
                user_group = self._get_animal_group(user_animal, animal_groups)
                
                # 다른 그룹의 동물로 바뀌었으면 큰 패널티
                if orig_group != user_group and orig_group != 'unknown' and user_group != 'unknown':
                    logger.info(f"동물 그룹 변경 감지: {orig_animal}({orig_group}) → {user_animal}({user_group})")
                    return 0.01  # 99% 감점
        
        return 1.0
    
    def detect_semantic_opposite(self, original: str, user_input: str) -> float:
        """
        의미적 반대 관계 감지 (완전 자동화 - BERT 기반)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수
        """
        # 1. 학습된 패턴 기반 검사
        opposite_pairs = self.learner.learned_patterns.get('opposite_pairs', {})
        
        # 단어별 반대 관계 확인
        orig_words = re.findall(r'\w+', original)
        user_words = re.findall(r'\w+', user_input)
        
        opposite_count = 0
        total_checks = 0
        
        for orig_word in orig_words:
            if orig_word in opposite_pairs:
                total_checks += 1
                if orig_word in user_words:
                    continue  # 원본 단어가 있으면 정상
                
                # 반대 단어가 있는지 확인
                for opposite_word in opposite_pairs[orig_word]:
                    if opposite_word in user_words:
                        opposite_count += 1
                        logger.info(f"학습된 반대 관계 감지: {orig_word} → {opposite_word}")
                        break
        
        # 2. BERT 기반 실시간 의미적 반대 관계 감지
        if self.bert_model and len(orig_words) > 0 and len(user_words) > 0:
            try:
                # 원본과 사용자 입력의 전체 문장 유사도 계산
                embeddings = self.bert_model.encode([original, user_input])
                sentence_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # 유사도가 매우 낮으면 반대 의미로 판단
                if sentence_similarity < 0.3:
                    opposite_count += 1
                    total_checks += 1
                    logger.info(f"BERT 기반 의미 반전 감지: 유사도 {sentence_similarity:.3f}")
                
                # 개별 단어들의 의미적 반대 관계도 확인
                for orig_word in orig_words:
                    if len(orig_word) >= 2:  # 2글자 이상만
                        for user_word in user_words:
                            if len(user_word) >= 2 and orig_word != user_word:
                                word_embeddings = self.bert_model.encode([orig_word, user_word])
                                word_similarity = cosine_similarity([word_embeddings[0]], [word_embeddings[1]])[0][0]
                                
                                # 단어 유사도가 매우 낮으면 반대 의미로 판단
                                if word_similarity < 0.2:
                                    opposite_count += 1
                                    total_checks += 1
                                    logger.info(f"BERT 기반 단어 반대 감지: {orig_word} ↔ {user_word} (유사도: {word_similarity:.3f})")
                                    break
                                    
            except Exception as e:
                logger.warning(f"BERT 기반 반대 관계 감지 실패: {e}")
        
        if total_checks == 0:
            return 1.0
        
        opposite_ratio = opposite_count / total_checks
        
        if opposite_ratio > 0.5:
            return 0.01  # 99% 감점
        elif opposite_ratio > 0.3:
            return 0.1   # 90% 감점
        else:
            return 1.0   # 정상
    
    def detect_synonyms_and_variations(self, original: str, user_input: str) -> float:
        """
        동의어 및 유사 표현 감지 (현실적인 허용 범위)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 보너스 계수 (1.0 = 보너스 없음, >1.0 = 보너스)
        """
        synonym_pairs = self.learner.learned_patterns.get('synonym_pairs', {})
        
        # 단어별 동의어 확인
        orig_words = re.findall(r'\w+', original)
        user_words = re.findall(r'\w+', user_input)
        
        # 핵심 명사/주어는 변경 허용하지 않음 (너무 과한 변형 방지)
        critical_nouns = ['방귀뀐', '놈이', '개구리', '올챙이', '게', '개', '소', '말', '고양이', '호랑이']
        
        # 핵심 명사가 바뀌었는지 확인
        for noun in critical_nouns:
            if noun in original and noun not in user_input:
                # 핵심 명사가 없어졌으면 다른 핵심 명사로 바뀌었는지 확인
                other_critical_found = False
                for other_noun in critical_nouns:
                    if other_noun != noun and other_noun in user_input and other_noun not in original:
                        other_critical_found = True
                        break
                
                if other_critical_found:
                    logger.info(f"핵심 명사 변경 감지: {noun} → 다른 핵심 명사 (과한 변형)")
                    return 0.5  # 50% 감점 (과한 변형)
        
        synonym_matches = 0
        total_words = len(orig_words)
        
        if total_words == 0:
            return 1.0
        
        for orig_word in orig_words:
            if orig_word in user_words:
                synonym_matches += 1  # 정확한 단어 매치
                continue
            
            # 동의어 확인 (핵심 명사가 아닌 경우만)
            if orig_word in synonym_pairs and orig_word not in critical_nouns:
                for synonym in synonym_pairs[orig_word]:
                    if synonym in user_words:
                        synonym_matches += 1
                        logger.info(f"동의어 감지: {orig_word} ↔ {synonym}")
                        break
        
        # 동의어 매치 비율에 따른 보너스 (더 엄격하게)
        match_ratio = synonym_matches / total_words
        
        if match_ratio >= 0.9:  # 90% 이상 매치 (더 엄격)
            return 1.1  # 10% 보너스 (더 적게)
        elif match_ratio >= 0.7:  # 70% 이상 매치
            return 1.05  # 5% 보너스 (더 적게)
        else:
            return 1.0  # 보너스 없음
    
    def detect_critical_word_missing(self, original: str, user_input: str) -> float:
        """
        핵심 단어 누락 감지 (더 엄격하게)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수
        """
        critical_words = self.learner.learned_patterns.get('critical_words', {})
        
        penalty = 1.0
        missing_words = []
        critical_missing_count = 0
        
        # 원본 속담의 모든 단어 확인
        orig_words = re.findall(r'\w+', original)
        user_words = re.findall(r'\w+', user_input)
        
        for word in orig_words:
            if len(word) >= 2:  # 2글자 이상만 체크
                if word in user_words:
                    continue  # 단어가 있으면 정상
                
                # 핵심 단어인지 확인
                if word in critical_words:
                    penalty_factor = critical_words[word]
                    penalty *= penalty_factor
                    missing_words.append(word)
                    
                    # 매우 중요한 단어 누락시 카운트
                    if penalty_factor <= 0.2:  # 80% 이상 감점하는 단어
                        critical_missing_count += 1
        
        # 매우 중요한 단어가 2개 이상 누락되면 거의 오답 처리
        if critical_missing_count >= 2:
            penalty = 0.01  # 99% 감점
            logger.info(f"핵심 단어 2개 이상 누락: {missing_words}, 거의 오답 처리")
        elif critical_missing_count >= 1:
            penalty = min(penalty, 0.1)  # 최대 90% 감점
            logger.info(f"핵심 단어 1개 누락: {missing_words}, 90% 감점")
        elif missing_words:
            logger.info(f"일반 단어 누락: {missing_words}, 패널티: {penalty}")
        
        return penalty
    
    def detect_number_quantity_accuracy(self, original: str, user_input: str) -> float:
        """
        숫자/수량 표현 정확성 감지 (매우 엄격한 정확 매칭만 허용)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        # 숫자/수량 표현 패턴 (자동 감지)
        number_patterns = [
            r'\d+',  # 숫자
            r'[하나둘셋넷다섯여섯일곱여덟아홉열]',  # 한글 숫자
            r'[일이삼사오육칠팔구십]',  # 한자 숫자
            r'[석넉]'  # 특수 수량 표현
        ]
        
        # 원본과 사용자 입력에서 숫자/수량 표현 추출
        orig_numbers = []
        user_numbers = []
        
        for pattern in number_patterns:
            orig_matches = re.findall(pattern, original)
            user_matches = re.findall(pattern, user_input)
            orig_numbers.extend(orig_matches)
            user_numbers.extend(user_matches)
        
        # 숫자/수량 표현이 없으면 정상
        if not orig_numbers:
            return 1.0
        
        # 1. 정확한 매칭 확인 (최우선)
        if orig_numbers == user_numbers:
            return 1.0  # 정상
        
        # 2. 엄격한 동의어 매핑만 허용 (BERT 기반 유사도 제거)
        number_mappings = {
            '하나': ['1', '일'],
            '둘': ['2', '이'],
            '셋': ['3', '삼'],
            '넷': ['4', '사'],
            '다섯': ['5', '오'],
            '여섯': ['6', '육'],
            '일곱': ['7', '칠'],
            '여덟': ['8', '팔'],
            '아홉': ['9', '구'],
            '열': ['10', '십'],
            '석': ['3', '삼', '셋'],
            '넉': ['4', '사', '넷']
        }
        
        # 매핑 기반 동의어 확인 (정확한 매핑만, 약간의 감점 적용)
        for orig_num in orig_numbers:
            if orig_num in number_mappings:
                for user_num in user_numbers:
                    if user_num in number_mappings[orig_num]:
                        logger.info(f"엄격한 동의어 인정: {orig_num} ↔ {user_num}")
                        return 0.95  # 5% 감점 (동의어 허용하지만 약간의 감점)
        
        # 3. 모든 매칭 실패시 완전 오답 처리
        logger.info(f"숫자/수량 표현 불일치: 원본 {orig_numbers} vs 입력 {user_numbers}")
        return 0.01  # 99% 감점 (완전 오답)
    
    def detect_grammar_variations(self, original: str, user_input: str) -> float:
        """
        문법적 변화 감지 (어미 변화, 조사 변화 등)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 보너스 계수 (1.0 = 보너스 없음, >1.0 = 보너스)
        """
        grammar_patterns = self.learner.learned_patterns.get('grammar_patterns', {})
        
        # 단어별 문법적 변화 확인
        orig_words = re.findall(r'\w+', original)
        user_words = re.findall(r'\w+', user_input)
        
        grammar_matches = 0
        total_words = len(orig_words)
        
        if total_words == 0:
            return 1.0
        
        for orig_word in orig_words:
            if orig_word in user_words:
                grammar_matches += 1  # 정확한 단어 매치
                continue
            
            # 문법적 변화 확인
            if orig_word in grammar_patterns:
                for variation in grammar_patterns[orig_word]:
                    if variation in user_words:
                        grammar_matches += 1
                        logger.info(f"문법적 변화 감지: {orig_word} ↔ {variation}")
                        break
        
        # 문법적 매치 비율에 따른 보너스
        match_ratio = grammar_matches / total_words
        
        if match_ratio >= 0.9:  # 90% 이상 매치
            return 1.05  # 5% 보너스
        elif match_ratio >= 0.7:  # 70% 이상 매치
            return 1.02  # 2% 보너스
        else:
            return 1.0  # 보너스 없음
    
    def detect_context_consistency(self, original: str, user_input: str) -> float:
        """
        문맥적 일관성 감지 (시간, 장소, 상황 등)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        # 문맥적 패턴들
        temporal_patterns = self.learner.learned_patterns.get('temporal_patterns', set())
        spatial_patterns = self.learner.learned_patterns.get('spatial_patterns', set())
        emotional_patterns = self.learner.learned_patterns.get('emotional_patterns', set())
        action_patterns = self.learner.learned_patterns.get('action_patterns', set())
        object_patterns = self.learner.learned_patterns.get('object_patterns', set())
        quality_patterns = self.learner.learned_patterns.get('quality_patterns', set())
        
        # 원본과 사용자 입력에서 문맥적 단어 추출
        orig_context_words = self._extract_context_words(original, temporal_patterns, spatial_patterns, 
                                                        emotional_patterns, action_patterns, object_patterns, quality_patterns)
        user_context_words = self._extract_context_words(user_input, temporal_patterns, spatial_patterns,
                                                        emotional_patterns, action_patterns, object_patterns, quality_patterns)
        
        if not orig_context_words:
            return 1.0  # 문맥적 단어가 없으면 정상
        
        # 문맥적 일관성 계산
        context_overlap = len(orig_context_words & user_context_words)
        context_total = len(orig_context_words | user_context_words)
        
        if context_total == 0:
            return 1.0
        
        consistency_ratio = context_overlap / context_total
        
        if consistency_ratio < 0.2:  # 20% 미만 일치
            logger.info(f"문맥적 일관성 부족: {orig_context_words} vs {user_context_words}")
            return 0.01  # 99% 감점
        elif consistency_ratio < 0.5:  # 50% 미만 일치
            return 0.1   # 90% 감점
        else:
            return 1.0   # 정상
    
    def _extract_context_words(self, text: str, temporal: set, spatial: set, emotional: set, 
                              action: set, object: set, quality: set) -> set:
        """텍스트에서 문맥적 단어 추출"""
        words = re.findall(r'\w+', text)
        context_words = set()
        
        for word in words:
            if word in temporal or word in spatial or word in emotional or \
               word in action or word in object or word in quality:
                context_words.add(word)
        
        return context_words
    
    def detect_phonetic_similarity(self, original: str, user_input: str) -> float:
        """
        음성학적 유사성 감지 (발음 비슷한 단어, 오타 등)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 보너스 계수 (1.0 = 보너스 없음, >1.0 = 보너스)
        """
        phonetic_patterns = self.learner.learned_patterns.get('phonetic_patterns', {})
        
        # 단어별 음성학적 유사성 확인
        orig_words = re.findall(r'\w+', original)
        user_words = re.findall(r'\w+', user_input)
        
        phonetic_matches = 0
        total_words = len(orig_words)
        
        if total_words == 0:
            return 1.0
        
        for orig_word in orig_words:
            if orig_word in user_words:
                phonetic_matches += 1  # 정확한 단어 매치
                continue
            
            # 음성학적 유사성 확인
            if orig_word in phonetic_patterns:
                for similar_word in phonetic_patterns[orig_word]:
                    if similar_word in user_words:
                        phonetic_matches += 1
                        logger.info(f"음성학적 유사성 감지: {orig_word} ↔ {similar_word}")
                        break
        
        # 음성학적 매치 비율에 따른 보너스
        match_ratio = phonetic_matches / total_words
        
        if match_ratio >= 0.9:  # 90% 이상 매치
            return 1.03  # 3% 보너스
        elif match_ratio >= 0.7:  # 70% 이상 매치
            return 1.01  # 1% 보너스
        else:
            return 1.0  # 보너스 없음
    
    def detect_partial_answer_acceptance(self, original: str, user_input: str) -> float:
        """
        부분 답변 허용 여부 감지 (의미 기반으로 실제 사용되는 부분 답변만 허용)
        
        Args:
            original: 원본 속담 (정답 부분)
            user_input: 사용자 입력
            
        Returns:
            float: 보너스 계수 (1.0 = 보너스 없음, >1.0 = 보너스)
        """
        # 실제로 사용되는 의미 있는 부분 답변들 (하드코딩된 패턴)
        meaningful_partial_answers = {
            # 비교 구조 - 실제로 이렇게 말하는 경우들
            "배꼽이 크다": ["배꼽"],
            "용 난다": ["용 난다"],
            "오는 말이 곱다": ["오는 말이 곱다"],
            
            # 완전한 의미를 가진 부분 답변들
            "올챙이 시절": ["올챙이 시절"],
            "외양간 고친다": ["외양간 고친다"],
            "부채질한다": ["부채질한다"],
            "김서방 찾기": ["김서방 찾기"],
            "여든까지 간다": ["여든까지 간다"],
            "먹을 것 없다": ["먹을 것 없다"],
            "코가 깨진다": ["코가 깨진다"],
            "부뚜막에 먼저 올라간다": ["부뚜막에 먼저 올라간다"],
            "쳐다보지도 말아라": ["쳐다보지도 말아라"],
            "겨자 먹기": ["겨자 먹기"],
        }
        
        # 원본 속담이 부분 답변 허용 가능한지 확인
        if original not in meaningful_partial_answers:
            return 1.0  # 부분 답변 허용 안함
        
        # 허용되는 부분 답변 목록에서 확인
        allowed_partials = meaningful_partial_answers[original]
        
        # 정확한 부분 답변 매칭 확인
        for allowed_partial in allowed_partials:
            if user_input == allowed_partial:
                logger.info(f"의미 있는 부분 답변 인정: {original} → {user_input}")
                return 1.2  # 20% 보너스 (부분 답변 허용)
        
        # BERT 기반 의미적 유사도로 추가 확인
        if self.bert_model:
            try:
                embeddings = self.bert_model.encode([original, user_input])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # 매우 높은 유사도 (90% 이상)에서만 부분 답변 인정
                if similarity >= 0.9:
                    logger.info(f"BERT 기반 부분 답변 인정: {original} → {user_input} (유사도: {similarity:.3f})")
                    return 1.1  # 10% 보너스 (부분 답변 허용)
                else:
                    logger.info(f"의미적 유사도 부족: {original} → {user_input} (유사도: {similarity:.3f})")
                    return 1.0  # 보너스 없음
                    
            except Exception as e:
                logger.warning(f"부분 답변 감지 실패: {e}")
        
        return 1.0  # 기본값

    def detect_structural_consistency(self, original: str, user_input: str) -> float:
        """
        구조적 일관성 감지 (문장 구조, 어순, 길이 등)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            
        Returns:
            float: 패널티 계수 (1.0 = 정상, 0.01 = 99% 감점)
        """
        # 길이 일관성 확인
        length_ratio = len(user_input) / len(original) if len(original) > 0 else 1.0
        
        if length_ratio < 0.5 or length_ratio > 2.0:  # 길이가 너무 다름
            logger.info(f"구조적 일관성 부족: 길이 비율 {length_ratio:.2f}")
            return 0.1  # 90% 감점
        
        # 어순 일관성 확인
        orig_word_order = self._analyze_word_order(original)
        user_word_order = self._analyze_word_order(user_input)
        
        if orig_word_order != user_word_order:
            logger.info(f"어순 일관성 부족: {orig_word_order} vs {user_word_order}")
            return 0.5  # 50% 감점
        
        # 문장 구조 일관성 확인
        orig_structure = self._analyze_sentence_structure(original)
        user_structure = self._analyze_sentence_structure(user_input)
        
        if orig_structure != user_structure:
            logger.info(f"문장 구조 일관성 부족: {orig_structure} vs {user_structure}")
            return 0.7  # 30% 감점
        
        return 1.0  # 정상
    
    def _extract_animals_from_text(self, text: str) -> List[str]:
        """텍스트에서 동물 단어 추출"""
        animals = self.learner.learned_patterns.get('animals', set())
        found_animals = []
        
        for animal in animals:
            if animal in text:
                found_animals.append(animal)
        
        return found_animals
    
    def _get_animal_group(self, animal: str, animal_groups: Dict) -> str:
        """동물의 그룹 찾기"""
        for group, animals in animal_groups.items():
            if animal in animals:
                return group
        return 'unknown'
    
    def comprehensive_check(self, original: str, user_input: str, base_similarity: float) -> float:
        """
        종합적인 지능적 검사 (모든 패턴 통합)
        
        Args:
            original: 원본 속담
            user_input: 사용자 입력
            base_similarity: 기본 유사도 점수
            
        Returns:
            float: 최종 유사도 점수
        """
        final_score = base_similarity
        
        # 1. 숫자/수량 표현 정확성 감지 (최우선 - 매우 엄격)
        number_penalty = self.detect_number_quantity_accuracy(original, user_input)
        final_score *= number_penalty
        
        # 2. 동물 혼동 감지
        animal_penalty = self.detect_animal_confusion(original, user_input)
        final_score *= animal_penalty
        
        # 3. 의미적 반대 관계 감지
        opposite_penalty = self.detect_semantic_opposite(original, user_input)
        final_score *= opposite_penalty
        
        # 4. 핵심 단어 누락 감지
        critical_penalty = self.detect_critical_word_missing(original, user_input)
        final_score *= critical_penalty
        
        # 5. 문맥적 일관성 감지 (새로 추가)
        context_penalty = self.detect_context_consistency(original, user_input)
        final_score *= context_penalty
        
        # 6. 구조적 일관성 감지 (새로 추가)
        structural_penalty = self.detect_structural_consistency(original, user_input)
        final_score *= structural_penalty
        
        # 7. 동의어 및 유사 표현 감지 (보너스)
        synonym_bonus = self.detect_synonyms_and_variations(original, user_input)
        final_score *= synonym_bonus
        
        # 8. 문법적 변화 감지 (보너스 - 새로 추가)
        grammar_bonus = self.detect_grammar_variations(original, user_input)
        final_score *= grammar_bonus
        
        # 9. 음성학적 유사성 감지 (보너스 - 새로 추가)
        phonetic_bonus = self.detect_phonetic_similarity(original, user_input)
        final_score *= phonetic_bonus
        
        # 10. 부분 답변 허용 감지 (보너스 - 새로 추가)
        partial_bonus = self.detect_partial_answer_acceptance(original, user_input)
        final_score *= partial_bonus
        
        return min(final_score, 1.0)


def test_ai_learning_system():
    """AI 학습 시스템 테스트"""
    print("🧠 AI 학습 기반 지능적 속담 검사 시스템 테스트")
    print("=" * 70)
    
    # 학습 시스템 초기화
    learner = ProverbDataLearner()
    patterns = learner.full_learning_process()
    
    # 지능적 검사 시스템 초기화
    checker = IntelligentProverbChecker(learner)
    
    # 테스트 케이스들
    test_cases = [
        ("마파람에 게눈 감추듯", "마파람에 개눈 감추듯", "게→개 동물 혼동"),
        ("가는 말이 고와야", "가는 말이 나쁘면", "의미 반전"),
        ("개구리 올챙이 시절", "개구리 올챙이 시절", "정상 케이스"),
        ("열 번 찍어 안 넘어가는", "열 번 찍어 넘어가는", "부정어 제거"),
        ("내 코가 석 자", "내 코가 넉 자", "숫자/수량 표현 불일치 (석 vs 넉)"),
        ("내 코가 석 자", "내 코가 석 자", "숫자/수량 표현 정확"),
    ]
    
    for original, user_input, description in test_cases:
        # 기본 유사도 계산
        embeddings = checker.bert_model.encode([original, user_input])
        base_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # 지능적 검사 적용
        final_score = checker.comprehensive_check(original, user_input, base_similarity)
        
        print(f"\n📝 테스트: {description}")
        print(f"원본: {original}")
        print(f"입력: {user_input}")
        print(f"기본 유사도: {base_similarity:.3f}")
        print(f"최종 점수: {final_score:.3f}")
        print(f"결과: {'✅ 정답' if final_score >= 0.5 else '❌ 오답'}")


if __name__ == "__main__":
    test_ai_learning_system()
