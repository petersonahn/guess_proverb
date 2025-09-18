import warnings
import os
import mysql.connector
from sentence_transformers import SentenceTransformer, util

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 로그 레벨 조정

def get_answers_from_db():
    """데이터베이스에서 answer 값들만 가져오기"""
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root1234',
        database='proverb_game'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM proverb")
    answers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return answers

def get_threshold_by_length(answer):
    """글자수에 따른 정교한 threshold 값들"""
    length = len(answer)
    
    # 매우 짧은 경우 (1-2글자): 한 글자 오타가 치명적
    if length <= 2:
        return 0.85
    
    # 짧은 경우 (3-4글자): 한 글자 오타가 매우 중요
    elif length <= 4:
        return 0.80
    
    # 중간 짧은 경우 (5-6글자): 한 글자 오타가 중요하지만 어느 정도 허용
    elif length <= 6:
        return 0.75
    
    # 중간 경우 (7-8글자): 한 글자 오타 허용도 증가
    elif length <= 8:
        return 0.70
    
    # 중간 긴 경우 (9-12글자): 한 글자 오타 허용도 더 증가
    elif length <= 12:
        return 0.65
    
    # 긴 경우 (13-16글자): 한 글자 오타 허용도 높음
    elif length <= 16:
        return 0.60
    
    # 매우 긴 경우 (17글자 이상): 한 글자 오타 허용도 매우 높음
    else:
        return 0.55

def test_similarity(answer, test_cases, threshold=None, use_penalty=True):
    """유사도 테스트 함수"""
    if threshold is None:
        threshold = get_threshold_by_length(answer)
    
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    answer_embedding = model.encode([answer])[0]
    
    print(f"\n=== '{answer}' (길이: {len(answer)}) 테스트 ===")
    print(f"현재 threshold: {threshold}")
    if use_penalty:
        print("⚠️  패널티 적용: 단어순서바꾸기, 글자순서바꾸기 등에 유사도 감점")
    print("-" * 50)
    
    for test_input, description in test_cases:
        test_embedding = model.encode([test_input])[0]
        similarity = util.cos_sim(answer_embedding, test_embedding)[0].item()
        
        # 패널티 적용
        original_similarity = similarity
        if use_penalty:
            similarity = apply_penalty(similarity, description, answer, test_input)
        
        is_correct = similarity >= threshold
        
        status = "✓ 정답" if is_correct else "✗ 오답"
        # 패널티가 적용된 경우 원래 유사도 표시, 적용되지 않은 경우는 표시 안함
        penalty_info = f" (원래: {original_similarity:.3f})" if use_penalty and similarity != original_similarity else ""
        print(f"{status} | {description:20} | '{test_input}' | 유사도: {similarity:.3f}{penalty_info}")


def apply_penalty(similarity, description, answer, test_input):
    """특정 패턴에 대해 유사도에 패널티 적용 (고정 패널티)"""
    original_similarity = similarity
    
    # 1. 단어 순서 바꾸기 - 큰 패널티 (의미가 완전히 바뀜)
    if "단어 순서 바꾸기" in description:
        similarity *= 0.3  # 70% 감점
    
    # 2. 글자 순서 바꾸기 - 큰 패널티 (의미 변화 큼)
    elif "순서 바꾸기" in description:
        similarity *= 0.4  # 60% 감점
    
    # 3. 조사 변경 - 중간 패널티 (의미 변화 적음)
    elif "조사 변경" in description:
        similarity *= 0.9  # 10% 감점
    
    # 4. 글자 추가/제거 - 중간 패널티
    elif "글자 추가" in description or "글자 제거" in description:
        similarity *= 0.6  # 40% 감점
    
    # 5. 오타 - 작은 패널티
    elif "오타" in description:
        similarity *= 0.8  # 20% 감점
    
    # 6. 동의어 교체 - 작은 패널티 (의미는 비슷하므로)
    elif "동의어 교체" in description:
        similarity *= 0.95  # 5% 감점
    
    # 7. 띄어쓰기 관련 - 패널티 없음 (의미 변화 없음)
    elif "띄어쓰기" in description:
        pass  # 패널티 없음
    
    return similarity

def generate_test_cases(answer):
    """답안에 따른 다양한 테스트 케이스 생성"""
    test_cases = []
    
    # 1. 정확한 답안
    test_cases.append((answer, "정확한 답안"))
    
    # 2. 띄어쓰기 관련
    if len(answer) >= 2:
        # 중간에 띄어쓰기 추가
        for i in range(1, len(answer)):
            test_input = answer[:i] + " " + answer[i:]
            test_cases.append((test_input, f"띄어쓰기 추가({i}번째)"))
        
        # 띄어쓰기 제거 (띄어쓰기가 있는 경우)
        if " " in answer:
            test_input = answer.replace(" ", "")
            test_cases.append((test_input, "띄어쓰기 제거"))
    
    # 3. 글자 오타 (자주 틀리는 글자들)
    common_typos = {
        'ㅏ': ['ㅓ', 'ㅗ', 'ㅜ'],
        'ㅓ': ['ㅏ', 'ㅗ', 'ㅜ'],
        'ㅗ': ['ㅏ', 'ㅓ', 'ㅜ'],
        'ㅜ': ['ㅏ', 'ㅓ', 'ㅗ'],
        'ㅣ': ['ㅔ', 'ㅐ'],
        'ㅔ': ['ㅣ', 'ㅐ'],
        'ㅐ': ['ㅣ', 'ㅔ'],
        'ㄱ': ['ㄲ', 'ㅋ'],
        'ㄷ': ['ㄸ', 'ㅌ'],
        'ㅂ': ['ㅃ', 'ㅍ'],
        'ㅅ': ['ㅆ', 'ㅈ'],
        'ㅈ': ['ㅉ', 'ㅅ'],
        'ㅎ': ['ㅋ', 'ㅌ', 'ㅍ'],
    }
    
    # 자음/모음 오타 생성
    for i, char in enumerate(answer):
        if char in common_typos:
            for typo in common_typos[char]:
                test_input = answer[:i] + typo + answer[i+1:]
                test_cases.append((test_input, f"자음/모음 오타({i}번째)"))
    
    # 4. 발음 비슷한 글자 오타
    similar_sounds = {
        '가': ['까', '카'],
        '나': ['다', '라'],
        '다': ['나', '라', '타'],
        '라': ['나', '다'],
        '마': ['바', '파'],
        '바': ['마', '파', '빠'],
        '사': ['자', '차', '싸'],
        '아': ['어', '오', '우'],
        '어': ['아', '오', '우'],
        '오': ['아', '어', '우'],
        '우': ['아', '어', '오'],
        '자': ['사', '차', '짜'],
        '차': ['사', '자', '카'],
        '카': ['가', '차'],
        '타': ['다', '파'],
        '파': ['마', '바', '타'],
        '하': ['카', '타', '파'],
    }
    
    for i, char in enumerate(answer):
        if char in similar_sounds:
            for similar in similar_sounds[char]:
                test_input = answer[:i] + similar + answer[i+1:]
                test_cases.append((test_input, f"발음 유사 오타({i}번째)"))
    
    # 5. 동의어/유사 표현
    synonyms = {
        '가난한': ['빈곤한', '어려운', '궁핍한'],
        '가는': ['떠나는', '출발하는', '이동하는'],
        '말이': ['말은', '말을'],
        '고와야': ['좋아야', '예쁘게', '착하게'],
        '오는': ['돌아오는', '들어오는', '도착하는'],
        '말도': ['말은', '말을'],
        '고와야': ['좋아야', '예쁘게'],
        '장날': ['시장날', '장터날'],
        '가재는': ['가재는', '가재도'],
        '게': ['게도', '게는'],
        '집': ['가정', '집안', '가족'],
        '날이': ['날은', '날을'],
        '고와야': ['좋아야', '예쁘게'],
        '오는': ['돌아오는', '들어오는'],
        '말도': ['말은', '말을'],
    }
    
    # 동의어로 단어 교체
    words = answer.split()
    for i, word in enumerate(words):
        if word in synonyms:
            for synonym in synonyms[word]:
                new_words = words.copy()
                new_words[i] = synonym
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"동의어 교체({word}→{synonym})"))
    
    # 6. 글자 추가/제거
    if len(answer) >= 2:
        # 글자 하나 추가 (중간에)
        for i in range(1, len(answer)):
            test_input = answer[:i] + "가" + answer[i:]
            test_cases.append((test_input, f"글자 추가({i}번째)"))
        
        # 글자 하나 제거
        for i in range(len(answer)):
            test_input = answer[:i] + answer[i+1:]
            test_cases.append((test_input, f"글자 제거({i}번째)"))
    
    # 7. 순서 바꾸기 (2글자 이상)
    if len(answer) >= 2:
        for i in range(len(answer) - 1):
            chars = list(answer)
            chars[i], chars[i+1] = chars[i+1], chars[i]
            test_input = ''.join(chars)
            test_cases.append((test_input, f"순서 바꾸기({i},{i+1})"))
    
    # 8. 조사 변경
    josa_changes = {
        '이': ['은', '가'],
        '은': ['이', '가'],
        '가': ['이', '은'],
        '을': ['를'],
        '를': ['을'],
        '에': ['에서', '에게'],
        '에서': ['에', '에게'],
        '에게': ['에', '에서'],
        '의': ['이', '가'],
        '와': ['과'],
        '과': ['와'],
    }
    
    for i, char in enumerate(answer):
        if char in josa_changes:
            for josa in josa_changes[char]:
                test_input = answer[:i] + josa + answer[i+1:]
                test_cases.append((test_input, f"조사 변경({i}번째)"))
    
    # 9. 띄어쓰기 패턴 변경
    if " " in answer:
        # 띄어쓰기 위치 변경
        words = answer.split()
        if len(words) >= 2:
            # 단어 순서 바꾸기
            reversed_words = words[::-1]
            test_input = ' '.join(reversed_words)
            test_cases.append((test_input, "단어 순서 바꾸기"))
    
    return test_cases

def interactive_test():
    """대화형 테스트 모드"""
    print("=== 유사도 테스트 도구 ===")
    print("속담의 뒷부분을 입력하면 다양한 오타 패턴으로 유사도를 테스트합니다.")
    print("종료하려면 'quit'을 입력하세요.\n")
    
    while True:
        answer = input("테스트할 속담 뒷부분을 입력하세요: ").strip()
        
        if answer.lower() == 'quit':
            break
            
        if not answer:
            continue
        
        # 테스트 옵션 선택
        print("\n테스트 옵션을 선택하세요:")
        print("1. 기본 테스트 (패널티 적용)")
        print("2. 패널티 없는 테스트")
        print("3. 비교 테스트 (패널티 있음/없음)")
        
        option = input("선택 (1-3, 엔터: 1): ").strip() or "1"
        
        # 테스트 케이스 생성
        test_cases = generate_test_cases(answer)
        
        if option == "1":
            # 기본 테스트 (패널티 적용)
            test_similarity(answer, test_cases, use_penalty=True)
        elif option == "2":
            # 패널티 없는 테스트
            test_similarity(answer, test_cases, use_penalty=False)
        elif option == "3":
            # 비교 테스트
            print("\n=== 패널티 적용 테스트 ===")
            test_similarity(answer, test_cases, use_penalty=True)
            print("\n=== 패널티 없는 테스트 ===")
            test_similarity(answer, test_cases, use_penalty=False)
        
        # 사용자 정의 threshold로 테스트
        print("\n" + "="*60)
        custom_threshold = input(f"사용자 정의 threshold 입력 (엔터: 현재값 {get_threshold_by_length(answer)} 유지): ").strip()
        
        if custom_threshold:
            try:
                custom_threshold = float(custom_threshold)
                print(f"\n=== 사용자 정의 threshold ({custom_threshold}) 테스트 ===")
                test_similarity(answer, test_cases, custom_threshold, use_penalty=True)
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
        
        print("\n" + "="*60)

def batch_test():
    """배치 테스트 - 데이터베이스의 answer 값들로 테스트"""
    try:
        answers = get_answers_from_db()
        print(f"=== 배치 테스트 (총 {len(answers)}개 속담) ===")
        
        # 길이별로 그룹화
        length_groups = {}
        for answer in answers:
            length = len(answer)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(answer)
        
        # 길이별로 정렬해서 출력
        for length in sorted(length_groups.keys()):
            print(f"\n--- {length}글자 속담들 ---")
            for answer in length_groups[length][:3]:  # 각 길이별로 최대 3개만 테스트
                test_cases = generate_test_cases(answer)
                test_similarity(answer, test_cases)
                print("\n" + "="*60)
                
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        print("미리 정의된 속담들로 테스트합니다.")
        
        # 데이터베이스 연결 실패 시 기본 속담들로 테스트
        test_proverbs = [
            "가는 말이 고와야",  # 6글자
            "가는 날이",  # 4글자
            "가재는",  # 3글자
            "가난한 집",  # 4글자
            "가는 말",  # 3글자
            "가랑비에",  # 4글자
            "가재는 게",  # 4글자
            "가는 날이 장날",  # 8글자
            "가난한 집에",  # 5글자
            "가는 말이 고와야 오는 말도",  # 13글자
        ]
        
        print("=== 배치 테스트 ===")
        for proverb in test_proverbs:
            test_cases = generate_test_cases(proverb)
            test_similarity(proverb, test_cases)
            print("\n" + "="*80)

if __name__ == "__main__":
    print("유사도 테스트 도구")
    print("1. 대화형 테스트")
    print("2. 배치 테스트")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        interactive_test()
    elif choice == "2":
        batch_test()
    else:
        print("잘못된 선택입니다. 대화형 테스트를 시작합니다.")
        interactive_test()
