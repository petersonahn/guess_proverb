import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time
import re
from typing import List, Tuple
from functools import lru_cache

# ===== 상수 정의 =====
# 속담 어미 변화 패턴
PROVERB_ENDING_CHANGES = {
    '다': ['는다', '한다', '된다', '라', '디', '드'],
    '는다': ['다', 'ㄴ다', '는구나'],
    '한다': ['다', '해', '한다고'],
    '된다': ['다', '되나', '돼'],
    '기': ['키', '끼', '지'],
    '라': ['다', '어라', '아라'],
    '듯': ['듯이', '것처럼'],
    '모른다': ['몰라', '모르네', '모른다고'],
    '않는다': ['안 한다', '않아', '안돼'],
    '없다': ['없어', '없네', '없다고'],
    '편이라': ['편', '편이야', '편이다'],
}

# 속담 동의어/유사 표현
PROVERB_SYNONYMS = {
    '편이라': ['편', '편이다', '편이야'],
    '남 말한다': ['남 나무란다', '남 탓한다'],
    '올챙이 시절': ['올챙이 적', '올챙이 때'],
    '개도 안 때린다': ['개도 안 건드린다', '개도 건드리지 않는다'],
    '나무에서 떨어진다': ['나무에서 떨어질 때가 있다', '나무에서도 떨어진다'],
    '도구를 탓하지 않는다': ['연장을 탓하지 않는다', '도구 탓 안 한다'],
    '댓돌을 뚫는다': ['바위를 뚫는다', '돌을 뚫는다'],
    '우물 판다': ['우물을 판다', '우물 파기'],
    '별 따기': ['별따기', '별 따는 것'],
    '개구리': ['개굴이', '개구리가'],
    '진주목걸이': ['진주 목걸이', '목걸이'],
    '식후경': ['식후에 경치', '밥 먹고 경치'],
}

# 한글 오타 패턴
KOREAN_TYPOS = {
    '가': ['까', '카', '나'], '나': ['다', '라', '가'], '다': ['나', '라', '타', '따'],
    '라': ['나', '다', '아'], '마': ['바', '파', '나'], '바': ['마', '파', '빠', '다'],
    '사': ['자', '차', '싸', '따'], '자': ['사', '차', '짜', '가'], '차': ['사', '자', '카', '타'],
    '카': ['가', '차', '타'], '타': ['다', '파', '차', '카'], '파': ['마', '바', '타', '하'],
    '하': ['카', '타', '파', '가'], '아': ['어', '오', '우'], '어': ['아', '오', '으'],
    '오': ['아', '어', '우'], '우': ['오', '으', '아'], '으': ['어', '우', '이'],
    '이': ['으', '어', '에'], '에': ['이', '애', '어'], '애': ['에', '이', '아'],
}

# 조사 변경 패턴
JOSA_CHANGES = {
    '이': ['가', '은'], '가': ['이', '은'], '은': ['이', '가'],
    '을': ['를'], '를': ['을'], '에': ['에서', '로'],
    '에서': ['에', '로'], '로': ['에', '으로'], '과': ['와', '하고'],
    '와': ['과', '하고'], '도': ['만', '까지'], '만': ['도', '까지'],
}

# 숫자 변환 매핑
NUMBER_MAPPINGS = [
    ('석 자', '3자'), ('석자', '3자'), ('석', '3'),
    ('세 ', '3 '), ('세살', '3살'),
    ('열 ', '10 '), ('열번', '10번'),
    ('천 리', '1000리'), ('천리', '1000리'), ('천', '1000'),
    ('한 걸음', '1걸음'), ('한걸음', '1걸음')
]

# 부정 표현 패턴
NEGATION_CORE_PATTERNS = [
    ('줄모른다', ['다', '는다', '한다']),
    ('지않는다', ['다', '는다', '한다']), 
    ('지못한다', ['다', '는다', '한다']),
    ('안된다', ['된다']),
    ('못한다', ['한다']),
    ('안때린다', ['때린다', '건드린다']),
    ('안넘어가는', ['넘어가는', '넘어간다']),
    ('없다', ['있다', '많다'])
]

# 핵심 단어 분류
NEGATION_WORDS = ['줄', '모른다', '않는다', '못한다', '안', '못', '없다']
ACTION_WORDS = ['간다', '온다', '난다', '된다', '뚫는다', '판다', '떨어진다', '기운다']
KEY_NOUNS = ['개구리', '올챙이', '도구', '연장', '댓돌', '바위', '우물', '별', '진주', '목걸이']
NUMBER_WORDS = ['석', '세', '열', '천', '한']
PARTICLES = ['이', '가', '은', '는', '을', '를', '에', '에서', '과', '와', '도', '만']

# 특수 패턴 (대조/조건 구조)
SPECIAL_PATTERNS = [
    ('달면', '쓰면'), ('가는', '오는'), ('윗물', '아랫물'),
    ('차면', '기운다'), ('익을수록', '숙인다'),
]

# 의미 변화 접두사 패턴 (새로 추가)
MEANING_CRITICAL_PREFIXES = {
    '개': ['개살구', '개똥', '개소리', '개구리'],  # 부정적 의미 또는 특별한 의미의 접두사
    '헛': ['헛소리', '헛일', '헛수고'],           # 무의미함을 나타내는 접두사
    '가짜': ['가짜'],                          # 가짜를 의미하는 접두사
    '빈': ['빈수레', '빈깡통'],                 # 빈 것을 의미하는 접두사
}

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
    속담 길이에 따른 세밀한 임계값 설정 (캐시 적용)
    
    핵심 원리:
    - 짧은 속담: 한 글자 실수가 큰 영향 → threshold 낮춤 (관대하게)
    - 긴 속담: 한두 글자 실수는 전체 맥락에서 허용 → threshold 높임 (엄격하게)
    - 5글자 이하는 글자별로 세밀하게 차등 적용
    """
    length = len(answer)
    
    # 임계값 매핑 테이블 사용으로 성능 개선
    threshold_map = [
        (2, 0.25), (3, 0.30), (4, 0.35), (5, 0.40),
        (6, 0.62), (8, 0.65), (10, 0.68),
        (15, 0.72), (20, 0.76), (25, 0.80), (30, 0.83)
    ]
    
    for max_length, threshold in threshold_map:
        if length <= max_length:
            return threshold
    
    return 0.85  # 매우 긴 속담 (30글자 이상)

def apply_penalty(similarity: float, description: str, answer: str, test_input: str) -> float:
    """
    속담 특성을 반영한 정교한 패널티 시스템
    실제 사용자 실수 패턴과 의미 변화 정도를 고려
    """
    # 패널티 규칙을 딕셔너리로 구조화하여 성능 개선
    penalty_rules = {
        "단어 순서 바꾸기": 0.2,
        "인접 단어 순서 바꾸기": 0.4,
        "조사 변경": 0.98,
        "띄어쓰기": 0.98,
        "정확한 답안": 1.02,
    }
    
    # 기본 패널티 적용
    for pattern, multiplier in penalty_rules.items():
        if pattern in description:
            return similarity * multiplier
    
    # 어미 변화 처리
    if "어미 변화" in description:
        negation_patterns = ['모른다', '않는다', '없다']
        if any(neg in description for neg in negation_patterns):
            return similarity * 0.3  # 70% 감점
        else:
            return similarity * 0.6  # 40% 감점
    
    # 표현 변화/동의어 교체 처리
    if "표현 변화" in description or "동의어 교체" in description:
        number_patterns = ['3', '세', '석', '10', '열', '1000', '천']
        key_word_patterns = ['도구', '연장', '바위', '댓돌']
        
        if any(num in description for num in number_patterns) or "숫자 표현" in description:
            return similarity * 0.98  # 2% 감점만
        elif any(key in description for key in key_word_patterns):
            return similarity * 0.7  # 30% 감점
        else:
            return similarity * 0.8  # 20% 감점
    
    # 오타 처리
    if "오타" in description:
        end_positions = ['마지막', '끝', str(len(answer)-1), str(len(answer)-2)]
        if any(pos in description for pos in end_positions):
            return similarity * 0.6  # 40% 감점
        else:
            return similarity * 0.75  # 25% 감점
    
    # 글자 추가/제거 처리
    josa_list = ['가', '이', '은', '는', '을', '를']
    
    if "글자 추가" in description:
        if any(josa in description for josa in josa_list):
            return similarity * 0.9  # 10% 감점만
        else:
            return similarity * 0.7  # 30% 감점
    
    if "글자 제거" in description:
        if any(josa in description for josa in josa_list):
            return similarity * 0.9  # 10% 감점만
        elif "마지막" in description or str(len(answer)-1) in description:
            return similarity * 0.5  # 50% 감점
        else:
            return similarity * 0.65  # 35% 감점
    
    # 기타 패턴
    return similarity * 0.8  # 20% 감점

def _generate_spacing_cases(answer: str) -> List[Tuple[str, str]]:
    """띄어쓰기 관련 테스트 케이스 생성"""
    test_cases = []
    
    if len(answer) >= 2:
        # 띄어쓰기 추가 (자연스러운 위치에만)
        natural_spaces = []
        for i in range(1, len(answer)):
            char_before = answer[i-1]
            char_after = answer[i]
            if (char_before in '은는이가을를에서와과도만' or 
                char_after in '가나다라마바사아자차카타파하' or
                i == len(answer) // 2):
                natural_spaces.append(i)
        
        for i in natural_spaces[:3]:  # 최대 3개만
            test_input = answer[:i] + " " + answer[i:]
            test_cases.append((test_input, f"띄어쓰기 추가({i}번째)"))
            
        # 띄어쓰기 제거
        if " " in answer:
            test_input = answer.replace(" ", "")
            test_cases.append((test_input, "띄어쓰기 제거"))
    
    return test_cases

def _generate_ending_cases(answer: str) -> List[Tuple[str, str]]:
    """어미 변화 테스트 케이스 생성"""
    test_cases = []
    
    for ending, alternatives in PROVERB_ENDING_CHANGES.items():
        if answer.endswith(ending):
            for alt in alternatives:
                test_input = answer[:-len(ending)] + alt
                test_cases.append((test_input, f"어미 변화({ending}→{alt})"))
    
    return test_cases

def _generate_synonym_cases(answer: str) -> List[Tuple[str, str]]:
    """동의어/유사 표현 테스트 케이스 생성"""
    test_cases = []
    
    # 단어 단위 동의어 교체
    words = answer.split()
    for i, word in enumerate(words):
        if word in PROVERB_SYNONYMS:
            for synonym in PROVERB_SYNONYMS[word]:
                new_words = words.copy()
                new_words[i] = synonym
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"동의어 교체({word}→{synonym})"))
    
    # 부분 문자열 동의어 교체
    for original, synonyms in PROVERB_SYNONYMS.items():
        if original in answer:
            for synonym in synonyms:
                test_input = answer.replace(original, synonym)
                test_cases.append((test_input, f"표현 변화({original}→{synonym})"))
    
    return test_cases

def _generate_number_cases(answer: str) -> List[Tuple[str, str]]:
    """숫자 표현 변화 테스트 케이스 생성"""
    test_cases = []
    
    for korean, arabic in NUMBER_MAPPINGS:
        if korean in answer:
            test_input = answer.replace(korean, arabic)
            test_cases.append((test_input, f"숫자 표현 변화({korean}→{arabic})"))
        elif arabic in answer:
            test_input = answer.replace(arabic, korean)
            test_cases.append((test_input, f"숫자 표현 변화({arabic}→{korean})"))
    
    return test_cases

def _generate_typo_cases(answer: str) -> List[Tuple[str, str]]:
    """오타 테스트 케이스 생성"""
    test_cases = []
    
    # 오타 생성 (처음 3글자와 마지막 3글자만)
    for i, char in enumerate(answer):
        if i < 3 or i >= len(answer) - 3:  # 처음/끝 부분만
            if char in KOREAN_TYPOS:
                for typo in KOREAN_TYPOS[char][:2]:  # 최대 2개만
                    test_input = answer[:i] + typo + answer[i+1:]
                    test_cases.append((test_input, f"오타({i}번째:{char}→{typo})"))
    
    return test_cases

def _generate_josa_cases(answer: str) -> List[Tuple[str, str]]:
    """조사 변경 및 추가/제거 테스트 케이스 생성"""
    test_cases = []
    
    # 조사 변경
    for i, char in enumerate(answer):
        if char in JOSA_CHANGES:
            for josa in JOSA_CHANGES[char][:2]:  # 최대 2개만
                test_input = answer[:i] + josa + answer[i+1:]
                test_cases.append((test_input, f"조사 변경({i}번째:{char}→{josa})"))
    
    # 조사 추가 (단어 끝에)
    words = answer.split()
    if words:
        last_word = words[-1]
        for josa in ['가', '이', '를', '을']:
            if not last_word.endswith(josa):
                new_words = words.copy()
                new_words[-1] = last_word + josa
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"조사 추가({josa})"))
    
    # 조사 제거
    common_josa = ['가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '도', '만']
    if answer and answer[-1] in common_josa:
        test_input = answer[:-1]
        if test_input:  # 빈 문자열이 아닌 경우만
            test_cases.append((test_input, f"조사 제거({answer[-1]})"))
    
    return test_cases

def _generate_modification_cases(answer: str) -> List[Tuple[str, str]]:
    """글자 추가/제거 테스트 케이스 생성"""
    test_cases = []
    
    if len(answer) >= 3:
        # 중간 부분에 글자 추가
        middle = len(answer) // 2
        for i in [middle-1, middle, middle+1]:
            if 0 < i < len(answer):
                test_input = answer[:i] + "ㅇ" + answer[i:]
                test_cases.append((test_input, f"글자 추가({i}번째)"))
        
        # 끝부분 글자 제거
        for i in range(max(0, len(answer)-3), len(answer)):
            test_input = answer[:i] + answer[i+1:]
            if test_input:  # 빈 문자열 방지
                test_cases.append((test_input, f"글자 제거({i}번째)"))
    
    return test_cases

def _generate_word_order_cases(answer: str) -> List[Tuple[str, str]]:
    """단어 순서 바꾸기 테스트 케이스 생성"""
    test_cases = []
    
    if " " in answer:
        words = answer.split()
        if len(words) == 2:
            # 2단어면 순서 바꾸기
            test_input = words[1] + " " + words[0]
            test_cases.append((test_input, "단어 순서 바꾸기"))
        elif len(words) >= 3:
            # 3단어 이상이면 인접한 단어들만 바꾸기
            for i in range(len(words) - 1):
                new_words = words.copy()
                new_words[i], new_words[i+1] = new_words[i+1], new_words[i]
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"인접 단어 순서 바꾸기({i},{i+1})"))
    
    return test_cases

def generate_test_cases(answer: str) -> List[Tuple[str, str]]:
    """
    실제 84개 속담 데이터를 분석하여 최적화된 테스트 케이스 생성
    속담 특유의 패턴과 사용자 실수 유형을 반영 (모듈화 및 최적화)
    """
    test_cases = [(answer, "정확한 답안")]
    
    # 각 유형별 테스트 케이스 생성 함수들을 순차적으로 호출
    generators = [
        _generate_spacing_cases,
        _generate_ending_cases,
        _generate_synonym_cases,
        _generate_number_cases,
        _generate_typo_cases,
        _generate_josa_cases,
        _generate_modification_cases,
        _generate_word_order_cases,
    ]
    
    for generator in generators:
        test_cases.extend(generator(answer))
    
    return test_cases

def detect_meaning_reversal(original: str, user_input: str) -> float:
    """
    속담 길이별 의미 반전 감지 (최적화)
    짧은 속담일수록 의미 반전에 더 민감하게 반응
    """
    length = len(original)
    original_no_space = original.replace(' ', '')
    user_input_no_space = user_input.replace(' ', '')
    
    # 길이별 패널티 강도 매핑
    penalty_map = [(5, 0.05), (10, 0.1), (20, 0.15)]
    base_penalty = 0.2  # 기본값 (긴 속담)
    
    for max_len, penalty in penalty_map:
        if length <= max_len:
            base_penalty = penalty
            break
    
    # 1. 부정문 ↔ 긍정문 변환 감지
    for neg_pattern, pos_forms in NEGATION_CORE_PATTERNS:
        # 부정문 → 긍정문
        if neg_pattern in original_no_space and neg_pattern not in user_input_no_space:
            if any(user_input_no_space.endswith(pos_form) for pos_form in pos_forms):
                return base_penalty
        
        # 긍정문 → 부정문
        if neg_pattern in user_input_no_space and neg_pattern not in original_no_space:
            if any(original_no_space.endswith(pos_form) for pos_form in pos_forms):
                return base_penalty
    
    # 2. 짧은 속담 핵심 단어 변경 감지
    if length <= 5:
        key_chars = set(original_no_space[-2:])  # 마지막 2글자
        user_chars = set(user_input_no_space[-2:] if len(user_input_no_space) >= 2 else user_input_no_space)
        
        if len(key_chars & user_chars) == 0:  # 교집합이 없음
            return base_penalty * 0.5
    
    # 3. 중간 길이 속담 핵심 단어 변화 감지
    elif length <= 15:
        original_words = original.split()
        user_words = user_input.split()
        
        if len(original_words) >= 2 and len(user_words) >= 2:
            if original_words[0] != user_words[0] and len(original_words[0]) >= 2:
                orig_chars = set(original_words[0])
                user_chars = set(user_words[0])
                if len(orig_chars & user_chars) <= 1:
                    return base_penalty * 0.7
    
    return 1.0  # 패널티 없음


def detect_meaning_prefix_missing(original: str, user_input: str) -> float:
    """
    의미 변화 접두사 누락 감지 (새로 추가)
    '개살구'에서 '살구'만 입력하는 등의 케이스를 감지하여 큰 패널티 적용
    """
    penalty = 1.0
    
    for prefix, words_with_prefix in MEANING_CRITICAL_PREFIXES.items():
        for word in words_with_prefix:
            if word in original:
                # 접두사가 포함된 단어가 원문에 있는 경우
                suffix = word[len(prefix):]  # 접두사 제거한 부분
                
                # 사용자가 접두사 없이 뒷부분만 입력한 경우
                if suffix in user_input and prefix not in user_input:
                    penalty *= 0.1  # 90% 감점 (매우 엄격)
                    break
    
    return penalty


def detect_critical_word_missing(original: str, user_input: str) -> float:
    """
    실제 84개 속담 데이터 기반 핵심 단어 누락 감지 (최적화)
    속담별 핵심 의미 단어들을 정교하게 분류하여 차등 패널티 적용
    """
    penalty = 1.0
    
    # 단어 그룹별 패널티 매핑
    word_penalties = [
        (NEGATION_WORDS, 0.1),  # 90% 감점 (매우 엄격)
        (ACTION_WORDS, 0.4),    # 60% 감점
        (KEY_NOUNS, 0.6),       # 40% 감점
    ]
    
    # 1-3. 핵심 단어 누락 검사
    for word_list, penalty_factor in word_penalties:
        for word in word_list:
            if word in original and word not in user_input:
                penalty *= penalty_factor
    
    # 4. 숫자/수량 표현 - 변형 허용
    for word in NUMBER_WORDS:
        if word in original and word not in user_input:
            # 숫자는 다른 형태로 표현 가능하므로 관대하게
            if not any(alt in user_input for alt in ['3', '10', '1000', '1']):
                penalty *= 0.8  # 20% 감점만
    
    # 5. 조사/어미 누락 검사
    missing_particles = sum(1 for word in PARTICLES 
                          if word in original and word not in user_input)
    
    if missing_particles > 2:  # 조사 3개 이상 누락시에만 패널티
        penalty *= 0.9  # 10% 감점만
    
    # 6. 특수 패턴 (대조/조건 구조) 검사
    for word1, word2 in SPECIAL_PATTERNS:
        if word1 in original and word2 in original:
            # 대조/조건 구조에서 한쪽만 누락되면 큰 패널티
            if (word1 not in user_input) != (word2 not in user_input):  # XOR
                penalty *= 0.3  # 70% 감점
    
    return penalty


@lru_cache(maxsize=256)
def check_number_expression_similarity(original: str, user_input: str) -> bool:
    """
    숫자 표현 변환 유사도 체크 (한글 ↔ 아라비아 숫자) - 캐시 적용
    """
    # 정규화된 문자열 비교 (띄어쓰기 제거)
    orig_normalized = original.replace(' ', '').lower()
    user_normalized = user_input.replace(' ', '').lower()
    
    # NUMBER_MAPPINGS 상수 사용 (이미 모듈 레벨에 정의됨)
    for korean, arabic in NUMBER_MAPPINGS:
        korean_norm = korean.replace(' ', '').lower()
        arabic_norm = arabic.replace(' ', '').lower()
        
        # 양방향 변환 체크를 더 효율적으로
        conversions = [
            (korean_norm, arabic_norm),  # 한글 → 아라비아
            (arabic_norm, korean_norm)   # 아라비아 → 한글
        ]
        
        for from_pattern, to_pattern in conversions:
            if from_pattern in orig_normalized and to_pattern in user_normalized:
                orig_converted = orig_normalized.replace(from_pattern, to_pattern)
                if orig_converted == user_normalized:
                    return True
    
    return False


def enhanced_similarity_check(original: str, user_input: str, base_similarity: float) -> float:
    """
    속담 길이별 맞춤형 유사도 검사 (최적화)
    짧은 속담일수록 더 엄밀하게, 긴 속담일수록 맥락 중심으로 판단
    하이브리드 접근법으로 의미 변화 접두사, 최소 길이 비율, 첫 글자 중요도 모두 적용
    """
    length = len(original)
    final_score = base_similarity
    
    # 0. 숫자 표현 특별 처리 (우선적으로 체크)
    if check_number_expression_similarity(original, user_input):
        final_score *= 1.1  # 10% 보너스
    
    # 1. 의미 변화 접두사 누락 감지 (새로 추가 - 최우선)
    prefix_penalty = detect_meaning_prefix_missing(original, user_input)
    final_score *= prefix_penalty
    
    # 2. 의미 반전 감지
    meaning_penalty = detect_meaning_reversal(original, user_input)
    final_score *= meaning_penalty
    
    # 3. 핵심 단어 누락 감지
    word_penalty = detect_critical_word_missing(original, user_input)
    final_score *= word_penalty
    
    # 4. 최소 길이 비율 조건 (새로 추가 - 하이브리드 접근법)
    if len(original) <= 5:  # 짧은 속담에만 적용
        input_length_ratio = len(user_input) / len(original) if len(original) > 0 else 0
        
        # 숫자 표현 변환이 감지되면 길이 비율 패널티 완화
        if check_number_expression_similarity(original, user_input):
            if input_length_ratio < 0.50:  # 50% 미만에만 적용 (더 관대)
                final_score *= 0.70  # 30% 감점만
        else:
            if input_length_ratio < 0.65:  # 65% 미만이면 패널티 (기존 75%에서 완화)
                final_score *= 0.30  # 70% 감점
    
    # 5. 길이 차이 패널티 (기존 로직 유지)
    len_diff = abs(len(original) - len(user_input))
    len_ratio = len_diff / len(original) if len(original) > 0 else 0
    
    length_penalties = [
        (5, 0.2, 0.5),   # (max_length, ratio_threshold, penalty)
        (10, 0.3, 0.6),
        (float('inf'), 0.5, 0.7)  # 긴 속담
    ]
    
    for max_len, ratio_threshold, penalty in length_penalties:
        if length <= max_len:
            if len_ratio > ratio_threshold:
                final_score *= penalty
            break
    
    # 6. 완전 동일 보너스 (띄어쓰기 차이만)
    if original.replace(' ', '') == user_input.replace(' ', ''):
        final_score *= 1.05  # 5% 보너스
    
    # 7. 짧은 속담 특별 처리 (첫 글자 중요도 강화)
    if length <= 5 and user_input:
        if original[0] != user_input[0]:
            final_score *= 0.3  # 0.7 → 0.3으로 더 엄격하게 (70% 감점)
        if original[-1] != user_input[-1]:
            final_score *= 0.6  # 마지막 글자 다름 (어미 중요)
    
    # 8. 긴 속담 특별 처리  
    elif length > 20:
        original_words = set(original.split())
        user_words = set(user_input.split())
        
        if original_words and len(original_words & user_words) / len(original_words) > 0.7:
            final_score *= 1.1  # 키워드 70% 이상 일치시 보너스
    
    return final_score


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
    속담 유사도 검사 (모델 캐싱 적용)
    """
    model = _get_model()
    answer_embeddings = model.encode(answers)
    user_embedding = model.encode([user_input])[0]
    similarities = util.cos_sim(user_embedding, answer_embeddings)[0]
    max_idx = similarities.argmax()
    base_score = similarities[max_idx].item()
    matched_answer = answers[max_idx]
    
    # 향상된 유사도 검사 적용
    enhanced_score = enhanced_similarity_check(matched_answer, user_input, base_score)
    
    return enhanced_score >= threshold, matched_answer, enhanced_score

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