import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time
import re

def get_proverbs_from_db():
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

def get_threshold_by_length(answer):
    """
    속담 길이에 따른 세밀한 임계값 설정
    
    핵심 원리:
    - 짧은 속담: 한 글자 실수가 큰 영향 → threshold 낮춤 (관대하게)
    - 긴 속담: 한두 글자 실수는 전체 맥락에서 허용 → threshold 높임 (엄격하게)
    - 5글자 이하는 글자별로 세밀하게 차등 적용
    """
    length = len(answer)
    
    # 초단문 속담들 - 한 글자 한 글자가 매우 중요하지만 띄어쓰기는 관대하게
    if length == 2:
        return 0.25  # 매우 관대 (예: 거의 없음)
    elif length == 3:
        return 0.30  # 매우 관대하되 의미 변화는 감지 (예: '식후경')
    elif length == 4:
        return 0.35  # 관대하되 의미 변화는 감지 (예: '석 자다' 띄어쓰기 허용)
    elif length == 5:
        return 0.40  # 약간 관대 (예: '용 난다', '게 편이라')
        
    # 짧은 속담 (6-10글자) - 여전히 관대
    elif length <= 6:
        return 0.62  # 예: '석 자다', '우물 판다'
    elif length <= 8:
        return 0.65  # 예: '별 따기', '개구리'
    elif length <= 10:
        return 0.68  # 예: '오는 말이 곱다'
        
    # 중간 길이 (11-20글자) - 표준 기준  
    elif length <= 15:
        return 0.72  # 예: '제 발 저리다', '맞들면 낫다'
    elif length <= 20:
        return 0.76  # 예: '배가 산으로 간다', '나무에서 떨어진다'
        
    # 긴 속담 (21-30글자) - 엄격한 기준
    elif length <= 25:
        return 0.80  # 예: '보따리 내놓으라 한다'
    elif length <= 30:
        return 0.83  # 예: '얌전한 고양이가 부뚜막에 먼저 올라간다'
        
    # 매우 긴 속담 (30글자 이상) - 매우 엄격
    else:
        return 0.85  # 복합 문장 수준
    
    # 설정 논리:
    # 2-5글자: 각각 다른 기준 (한 글자의 중요도 극대)
    # 6글자+: 점진적으로 엄격해짐 (맥락의 중요도 증가)

def apply_penalty(similarity, description, answer, test_input):
    """
    속담 특성을 반영한 정교한 패널티 시스템
    실제 사용자 실수 패턴과 의미 변화 정도를 고려
    """
    original_similarity = similarity
    
    # 1. 의미 완전 변화 - 매우 큰 패널티
    if "단어 순서 바꾸기" in description:
        similarity *= 0.2  # 80% 감점 (더 엄격)
    
    # 2. 어미 변화 - 속담에서 중요함
    elif "어미 변화" in description:
        # 부정 표현 변화는 더 큰 패널티
        if any(neg in description for neg in ['모른다', '않는다', '없다']):
            similarity *= 0.3  # 70% 감점
        else:
            similarity *= 0.6  # 40% 감점
    
    # 3. 표현 변화/동의어 교체 - 맥락에 따라 차등
    elif "표현 변화" in description or "동의어 교체" in description:
        # 숫자 표현 변화는 매우 허용적 (3→세, 석→3)
        if any(num in description for num in ['3', '세', '석', '10', '열', '1000', '천']) or "숫자 표현" in description:
            similarity *= 0.98  # 2% 감점만 (거의 동일하게 취급)
        # 핵심 단어 변화는 엄격
        elif any(key in description for key in ['도구', '연장', '바위', '댓돌']):
            similarity *= 0.7  # 30% 감점
        else:
            similarity *= 0.8  # 20% 감점
    
    # 4. 조사 변경 - 거의 패널티 없음 (의미 변화 거의 없음)
    elif "조사 변경" in description:
        similarity *= 0.98  # 2% 감점만
    
    # 5. 오타 - 위치와 중요도에 따라 차등
    elif "오타" in description:
        # 끝부분 오타는 더 중요 (어미 변화)
        if any(pos in description for pos in ['마지막', '끝', str(len(answer)-1), str(len(answer)-2)]):
            similarity *= 0.6  # 40% 감점
        else:
            similarity *= 0.75  # 25% 감점
    
    # 6. 글자 추가/제거 - 위치에 따라 차등
    elif "글자 추가" in description:
        # 조사 추가는 관대하게
        if any(josa in description for josa in ['가', '이', '은', '는', '을', '를']):
            similarity *= 0.9  # 10% 감점만
        else:
            similarity *= 0.7  # 30% 감점
    elif "글자 제거" in description:
        # 조사 제거는 관대하게
        if any(josa in description for josa in ['가', '이', '은', '는', '을', '를']):
            similarity *= 0.9  # 10% 감점만
        # 끝부분 제거는 더 심각 (어미 손실)
        elif "마지막" in description or str(len(answer)-1) in description:
            similarity *= 0.5  # 50% 감점
        else:
            similarity *= 0.65  # 35% 감점
    
    # 7. 인접 단어 순서 바꾸기 - 중간 패널티
    elif "인접 단어 순서 바꾸기" in description:
        similarity *= 0.4  # 60% 감점
    
    # 8. 띄어쓰기 관련 - 패널티 거의 없음 (가장 흔한 실수)
    elif "띄어쓰기" in description:
        similarity *= 0.98  # 2% 감점만
    
    # 9. 정확한 답안 - 보너스
    elif "정확한 답안" in description:
        similarity *= 1.02  # 2% 보너스
    
    # 10. 기타 알 수 없는 패턴 - 기본 패널티
    else:
        similarity *= 0.8  # 20% 감점
    
    return similarity

def generate_test_cases(answer):
    """
    실제 84개 속담 데이터를 분석하여 최적화된 테스트 케이스 생성
    속담 특유의 패턴과 사용자 실수 유형을 반영
    """
    test_cases = []
    
    # 1. 정확한 답안
    test_cases.append((answer, "정확한 답안"))
    
    # 2. 띄어쓰기 관련 (가장 흔한 실수)
    if len(answer) >= 2:
        # 띄어쓰기 추가 (자연스러운 위치에만)
        natural_spaces = []
        for i in range(1, len(answer)):
            char_before = answer[i-1]
            char_after = answer[i]
            # 자연스러운 띄어쓰기 위치 판단
            if (char_before in '은는이가을를에서와과도만' or 
                char_after in '가나다라마바사아자차카타파하' or
                i == len(answer) // 2):  # 중간 지점
                natural_spaces.append(i)
        
        for i in natural_spaces[:3]:  # 최대 3개만
            test_input = answer[:i] + " " + answer[i:]
            test_cases.append((test_input, f"띄어쓰기 추가({i}번째)"))
            
        # 띄어쓰기 제거
        if " " in answer:
            test_input = answer.replace(" ", "")
            test_cases.append((test_input, "띄어쓰기 제거"))
    
    # 3. 속담 특화 어미 변화 (실제 데이터 기반)
    proverb_ending_changes = {
        # 가장 흔한 패턴들
        '다': ['는다', '한다', '된다', '라', '디', '드'],
        '는다': ['다', 'ㄴ다', '는구나'],
        '한다': ['다', '해', '한다고'],
        '된다': ['다', '되나', '돼'],
        '기': ['키', '끼', '지'],
        '라': ['다', '어라', '아라'],
        '듯': ['듯이', '것처럼'],
        # 부정 표현 변화
        '모른다': ['몰라', '모르네', '모른다고'],
        '않는다': ['안 한다', '않아', '안돼'],
        '없다': ['없어', '없네', '없다고'],
        # 명사형 변화
        '편이라': ['편', '편이야', '편이다'],
        '개구리': ['개굴이', '개구리가'],
    }
    
    for ending, alternatives in proverb_ending_changes.items():
        if answer.endswith(ending):
            for alt in alternatives:
                test_input = answer[:-len(ending)] + alt
                test_cases.append((test_input, f"어미 변화({ending}→{alt})"))
    
    # 4. 속담 특화 동의어/유사 표현 (실제 데이터 확장)
    proverb_synonyms = {
        # 실제 속담에서 발견된 패턴들
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
        '석 자다': ['3자다', '석자', '석 자'],
        '천 리': ['1000리', '천리', '멀리'],
        # 숫자 표현 변화 (더 포괄적으로)
        '세 살': ['3살', '세살'],
        '열 번': ['10번', '열번'],
        '한 걸음': ['1걸음', '한걸음'],
        '석': ['3', '석자'],
        '석 자': ['3자', '석자'],
        '천': ['1000', '천리'],
        '천 리': ['1000리', '천리'],
    }
    
    # 동의어 교체 (단어 단위)
    words = answer.split()
    for i, word in enumerate(words):
        if word in proverb_synonyms:
            for synonym in proverb_synonyms[word]:
                new_words = words.copy()
                new_words[i] = synonym
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"동의어 교체({word}→{synonym})"))
    
    # 부분 문자열 동의어 교체 (더 적극적으로)
    for original, synonyms in proverb_synonyms.items():
        if original in answer:
            for synonym in synonyms:
                test_input = answer.replace(original, synonym)
                test_cases.append((test_input, f"표현 변화({original}→{synonym})"))
    
    # 숫자 표현 특별 처리 (한글 ↔ 아라비아 숫자)
    number_mappings = [
        ('석 자', '3자'), ('석자', '3자'), ('석', '3'),
        ('세 ', '3 '), ('세살', '3살'),
        ('열 ', '10 '), ('열번', '10번'),
        ('천 리', '1000리'), ('천리', '1000리'), ('천', '1000'),
        ('한 걸음', '1걸음'), ('한걸음', '1걸음')
    ]
    
    for korean, arabic in number_mappings:
        if korean in answer:
            test_input = answer.replace(korean, arabic)
            test_cases.append((test_input, f"숫자 표현 변화({korean}→{arabic})"))
        elif arabic in answer:
            test_input = answer.replace(arabic, korean)
            test_cases.append((test_input, f"숫자 표현 변화({arabic}→{korean})"))
    
    # 5. 속담 특화 오타 패턴 (한글 특성 반영)
    korean_typos = {
        # 자주 틀리는 한글 패턴
        '가': ['까', '카', '나'],
        '나': ['다', '라', '가'],
        '다': ['나', '라', '타', '따'],
        '라': ['나', '다', '아'],
        '마': ['바', '파', '나'],
        '바': ['마', '파', '빠', '다'],
        '사': ['자', '차', '싸', '따'],
        '자': ['사', '차', '짜', '가'],
        '차': ['사', '자', '카', '타'],
        '카': ['가', '차', '타'],
        '타': ['다', '파', '차', '카'],
        '파': ['마', '바', '타', '하'],
        '하': ['카', '타', '파', '가'],
        # 모음 변화
        '아': ['어', '오', '우'],
        '어': ['아', '오', '으'],
        '오': ['아', '어', '우'],
        '우': ['오', '으', '아'],
        '으': ['어', '우', '이'],
        '이': ['으', '어', '에'],
        '에': ['이', '애', '어'],
        '애': ['에', '이', '아'],
    }
    
    # 오타 생성 (처음 3글자와 마지막 3글자만)
    for i, char in enumerate(answer):
        if i < 3 or i >= len(answer) - 3:  # 처음/끝 부분만
            if char in korean_typos:
                for typo in korean_typos[char][:2]:  # 최대 2개만
                    test_input = answer[:i] + typo + answer[i+1:]
                    test_cases.append((test_input, f"오타({i}번째:{char}→{typo})"))
    
    # 6. 조사 변경 (속담에서 자주 나타나는 조사들)
    josa_changes = {
        '이': ['가', '은'],
        '가': ['이', '은'],
        '은': ['이', '가'],
        '을': ['를'],
        '를': ['을'],
        '에': ['에서', '로'],
        '에서': ['에', '로'],
        '로': ['에', '으로'],
        '과': ['와', '하고'],
        '와': ['과', '하고'],
        '도': ['만', '까지'],
        '만': ['도', '까지'],
    }
    
    for i, char in enumerate(answer):
        if char in josa_changes:
            for josa in josa_changes[char][:2]:  # 최대 2개만
                test_input = answer[:i] + josa + answer[i+1:]
                test_cases.append((test_input, f"조사 변경({i}번째:{char}→{josa})"))
    
    # 7. 글자 추가/제거 (중요 부분만)
    if len(answer) >= 3:
        # 중간 부분에 글자 추가 (1-2개만)
        middle = len(answer) // 2
        for i in [middle-1, middle, middle+1]:
            if 0 < i < len(answer):
                test_input = answer[:i] + "ㅇ" + answer[i:]
                test_cases.append((test_input, f"글자 추가({i}번째)"))
        
        # 끝부분 글자 제거 (1-2개만)
        for i in range(max(0, len(answer)-3), len(answer)):
            test_input = answer[:i] + answer[i+1:]
            if test_input:  # 빈 문자열 방지
                test_cases.append((test_input, f"글자 제거({i}번째)"))
    
    # 8. 조사 추가/제거 테스트 케이스 (더 체계적으로)
    common_josa = ['가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '도', '만']
    
    # 조사 추가 (단어 끝에)
    words = answer.split()
    if words:
        last_word = words[-1]
        # 마지막 단어에 조사 추가
        for josa in ['가', '이', '를', '을']:
            if not last_word.endswith(josa):
                new_words = words.copy()
                new_words[-1] = last_word + josa
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"조사 추가({josa})"))
    
    # 조사 제거 (마지막 글자가 조사인 경우)
    if answer and answer[-1] in common_josa:
        test_input = answer[:-1]
        if test_input:  # 빈 문자열이 아닌 경우만
            test_cases.append((test_input, f"조사 제거({answer[-1]})"))

    # 9. 단어 순서 바꾸기 (띄어쓰기가 있는 경우만)
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

def detect_meaning_reversal(original, user_input):
    """
    속담 길이별 의미 반전 감지
    짧은 속담일수록 의미 반전에 더 민감하게 반응
    """
    length = len(original)
    original_no_space = original.replace(' ', '')
    user_input_no_space = user_input.replace(' ', '')
    
    # 부정 표현의 핵심 패턴들 정의
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
    
    # 길이별 패널티 강도 설정
    if length <= 5:
        # 매우 짧은 속담 - 의미 반전시 거의 완전히 다른 말
        base_penalty = 0.05  # 95% 감점
    elif length <= 10:
        # 짧은 속담 - 의미 반전 매우 중요
        base_penalty = 0.1   # 90% 감점
    elif length <= 20:
        # 중간 속담 - 의미 반전 중요
        base_penalty = 0.15  # 85% 감점
    else:
        # 긴 속담 - 전체 맥락에서 의미 반전 판단
        base_penalty = 0.2   # 80% 감점
    
    # 1. 부정문 → 긍정문 변환 감지
    for neg_pattern, pos_forms in NEGATION_CORE_PATTERNS:
        if neg_pattern in original_no_space:
            if neg_pattern not in user_input_no_space:
                for pos_form in pos_forms:
                    if user_input_no_space.endswith(pos_form):
                        return base_penalty
    
    # 2. 긍정문 → 부정문 변환 감지  
    for neg_pattern, pos_forms in NEGATION_CORE_PATTERNS:
        if neg_pattern in user_input_no_space and neg_pattern not in original_no_space:
            for pos_form in pos_forms:
                if original_no_space.endswith(pos_form):
                    return base_penalty
    
    # 3. 짧은 속담에서 핵심 단어 완전 변경 감지
    if length <= 5:
        # 핵심 단어가 완전히 바뀐 경우 (예: "식후경" → "식전경")
        key_chars = set(original_no_space[-2:])  # 마지막 2글자
        user_chars = set(user_input_no_space[-2:] if len(user_input_no_space) >= 2 else user_input_no_space)
        
        if len(key_chars & user_chars) == 0:  # 교집합이 없음
            return base_penalty * 0.5  # 추가 패널티
    
    # 4. 중간 길이 속담에서 핵심 단어 변화 감지 (예: "오는" → "가는")
    elif length <= 15:
        # 첫 단어나 핵심 동사가 바뀐 경우
        original_words = original.split()
        user_words = user_input.split()
        
        if len(original_words) >= 2 and len(user_words) >= 2:
            # 첫 번째 단어가 완전히 다른 경우 (예: "오는" → "가는")
            if original_words[0] != user_words[0] and len(original_words[0]) >= 2:
                # 완전히 다른 단어인지 확인 (한 글자도 겹치지 않음)
                orig_chars = set(original_words[0])
                user_chars = set(user_words[0])
                if len(orig_chars & user_chars) <= 1:  # 1글자 이하만 겹침
                    return base_penalty * 0.7  # 큰 패널티
    
    return 1.0  # 패널티 없음


def detect_critical_word_missing(original, user_input):
    """
    실제 84개 속담 데이터 기반 핵심 단어 누락 감지
    속담별 핵심 의미 단어들을 정교하게 분류하여 차등 패널티 적용
    """
    penalty = 1.0
    
    # 1. 부정 표현 - 가장 중요 (의미 완전 변화)
    NEGATION_WORDS = ['줄', '모른다', '않는다', '못한다', '안', '못', '없다']
    for word in NEGATION_WORDS:
        if word in original and word not in user_input:
            penalty *= 0.1  # 90% 감점 (매우 엄격)
    
    # 2. 핵심 동작/상태 단어 - 높은 중요도
    ACTION_WORDS = ['간다', '온다', '난다', '된다', '뚫는다', '판다', '떨어진다', '기운다']
    for word in ACTION_WORDS:
        if word in original and word not in user_input:
            penalty *= 0.4  # 60% 감점
    
    # 3. 핵심 명사 - 중간 중요도  
    KEY_NOUNS = ['개구리', '올챙이', '도구', '연장', '댓돌', '바위', '우물', '별', '진주', '목걸이']
    for word in KEY_NOUNS:
        if word in original and word not in user_input:
            penalty *= 0.6  # 40% 감점
    
    # 4. 숫자/수량 표현 - 중간 중요도 (변형 허용)
    NUMBER_WORDS = ['석', '세', '열', '천', '한']
    for word in NUMBER_WORDS:
        if word in original and word not in user_input:
            # 숫자는 다른 형태로 표현 가능하므로 관대하게
            if not any(alt in user_input for alt in ['3', '10', '1000', '1']):
                penalty *= 0.8  # 20% 감점만
    
    # 5. 조사/어미 - 낮은 중요도 (의미 변화 적음)
    PARTICLES = ['이', '가', '은', '는', '을', '를', '에', '에서', '과', '와', '도', '만']
    missing_particles = 0
    for word in PARTICLES:
        if word in original and word not in user_input:
            missing_particles += 1
    
    if missing_particles > 2:  # 조사 3개 이상 누락시에만 패널티
        penalty *= 0.9  # 10% 감점만
    
    # 6. 특수 패턴 - 속담별 맞춤 검사
    special_patterns = [
        # 대조 구조
        ('달면', '쓰면'),  # "달면 삼키고 쓰면 뱉는다"
        ('가는', '오는'),  # "가는 말이 고와야 오는 말이 곱다"  
        ('윗물', '아랫물'),  # "윗물이 맑아야 아랫물이 맑다"
        # 조건 구조
        ('차면', '기운다'),  # "달도 차면 기운다"
        ('익을수록', '숙인다'),  # "벼는 익을수록 고개를 숙인다"
    ]
    
    for word1, word2 in special_patterns:
        if word1 in original and word2 in original:
            # 대조/조건 구조에서 한쪽만 누락되면 큰 패널티
            if (word1 not in user_input) != (word2 not in user_input):  # XOR
                penalty *= 0.3  # 70% 감점
    
    return penalty


def check_number_expression_similarity(original, user_input):
    """
    숫자 표현 변환 유사도 체크 (한글 ↔ 아라비아 숫자)
    """
    # 숫자 변환 매핑
    number_conversions = [
        ('석 자', '3자'), ('석자', '3자'), ('석', '3'),
        ('세 ', '3 '), ('세살', '3살'), ('세', '3'),
        ('열 ', '10 '), ('열번', '10번'), ('열', '10'),
        ('천 리', '1000리'), ('천리', '1000리'), ('천', '1000'),
        ('한 걸음', '1걸음'), ('한걸음', '1걸음'), ('한', '1')
    ]
    
    # 정규화된 문자열 비교 (띄어쓰기 제거)
    orig_normalized = original.replace(' ', '').lower()
    user_normalized = user_input.replace(' ', '').lower()
    
    # 직접 매칭 확인
    for korean, arabic in number_conversions:
        korean_norm = korean.replace(' ', '').lower()
        arabic_norm = arabic.replace(' ', '').lower()
        
        # 원본에 한글 숫자, 입력에 아라비아 숫자
        if korean_norm in orig_normalized and arabic_norm in user_normalized:
            orig_converted = orig_normalized.replace(korean_norm, arabic_norm)
            if orig_converted == user_normalized:
                return True
        
        # 원본에 아라비아 숫자, 입력에 한글 숫자  
        if arabic_norm in orig_normalized and korean_norm in user_normalized:
            orig_converted = orig_normalized.replace(arabic_norm, korean_norm)
            if orig_converted == user_normalized:
                return True
    
    return False


def enhanced_similarity_check(original, user_input, base_similarity):
    """
    속담 길이별 맞춤형 유사도 검사
    짧은 속담일수록 더 엄밀하게, 긴 속담일수록 맥락 중심으로 판단
    """
    length = len(original)
    final_score = base_similarity
    
    # 0. 숫자 표현 특별 처리 (우선적으로 체크)
    if check_number_expression_similarity(original, user_input):
        final_score *= 1.1  # 10% 보너스
    
    # 1. 의미 반전 감지 (모든 길이에서 중요)
    meaning_penalty = detect_meaning_reversal(original, user_input)
    final_score *= meaning_penalty
    
    # 2. 핵심 단어 누락 감지 (길이별 차등 적용)
    word_penalty = detect_critical_word_missing(original, user_input)
    final_score *= word_penalty
    
    # 3. 길이 차이 패널티 (길이별 허용 범위 다름)
    len_diff = abs(len(original) - len(user_input))
    len_ratio = len_diff / len(original) if len(original) > 0 else 0
    
    if length <= 5:
        # 짧은 속담: 한 글자 차이도 큰 영향
        if len_ratio > 0.2:  # 20% 이상 차이
            final_score *= 0.5  # 50% 감점
    elif length <= 10:
        # 중간 속담: 적당한 허용
        if len_ratio > 0.3:  # 30% 이상 차이
            final_score *= 0.6  # 40% 감점
    else:
        # 긴 속담: 상당한 차이까지 허용
        if len_ratio > 0.5:  # 50% 이상 차이
            final_score *= 0.7  # 30% 감점
    
    # 4. 완전 동일 보너스 (띄어쓰기 차이만 있는 경우)
    if original.replace(' ', '') == user_input.replace(' ', ''):
        final_score *= 1.05  # 5% 보너스
    
    # 5. 짧은 속담 특별 처리
    if length <= 5:
        # 짧은 속담에서 첫 글자나 마지막 글자가 다르면 큰 패널티
        if original[0] != user_input[0] if user_input else True:
            final_score *= 0.7  # 첫 글자 다름
        if original[-1] != user_input[-1] if user_input else True:
            final_score *= 0.6  # 마지막 글자 다름 (어미 중요)
    
    # 6. 긴 속담 특별 처리  
    elif length > 20:
        # 긴 속담에서는 핵심 키워드 포함 여부가 더 중요
        original_words = set(original.split())
        user_words = set(user_input.split())
        
        if len(original_words & user_words) / len(original_words) > 0.7:
            final_score *= 1.1  # 키워드 70% 이상 일치시 보너스
    
    return final_score


def check_proverb(user_input, answers, threshold):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    answer_embeddings = model.encode(answers)
    user_embedding = model.encode([user_input])[0]
    similarities = util.cos_sim(user_embedding, answer_embeddings)[0]
    max_idx = similarities.argmax()
    base_score = similarities[max_idx].item()
    matched_answer = answers[max_idx]
    
    # 향상된 유사도 검사 적용
    enhanced_score = enhanced_similarity_check(matched_answer, user_input, base_score)
    
    if enhanced_score >= threshold:
        return True, matched_answer, enhanced_score
    else:
        return False, matched_answer, enhanced_score

if __name__ == "__main__":
    proverbs = get_proverbs_from_db()
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
        answers = [row[2] for row in proverbs]
        threshold = get_threshold_by_length(answer)
        print(f"문제: {question}")
        wrong_count = 0
        hint_shown = False
        problem_start = time.time()
        while True:
            # 제한시간 체크
            if time.time() - start_time >= total_time:
                print("\n제한시간 종료!\n")
                break
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
    print(f"\n게임 종료! 총 점수: {score}")