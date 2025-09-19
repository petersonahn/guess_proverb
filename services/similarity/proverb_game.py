import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time
import re

def get_proverbs_from_db():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='1234',
        database='proverb_game'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM proverb")
    proverbs = cursor.fetchall()  # [(id, question, answer, hint), ...]
    cursor.close()
    conn.close()
    return proverbs

def get_threshold_by_length(answer):
    length = len(answer)
    # 글자수가 적을수록 임계값이 작아지도록 반대로 설정
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

def apply_penalty(similarity, description, answer, test_input):
    original_similarity = similarity
    if "단어 순서 바꾸기" in description:
        similarity *= 0.3
    elif "순서 바꾸기" in description:
        similarity *= 0.4
    elif "조사 변경" in description:
        similarity *= 0.9
    elif "글자 추가" in description or "글자 제거" in description:
        similarity *= 0.6
    elif "오타" in description:
        similarity *= 0.8
    elif "동의어 교체" in description:
        similarity *= 0.95
    elif "띄어쓰기" in description:
        pass
    return similarity

def generate_test_cases(answer):
    test_cases = []
    test_cases.append((answer, "정확한 답안"))
    if len(answer) >= 2:
        for i in range(1, len(answer)):
            test_input = answer[:i] + " " + answer[i:]
            test_cases.append((test_input, f"띄어쓰기 추가({i}번째)"))
        if " " in answer:
            test_input = answer.replace(" ", "")
            test_cases.append((test_input, "띄어쓰기 제거"))
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
    for i, char in enumerate(answer):
        if char in common_typos:
            for typo in common_typos[char]:
                test_input = answer[:i] + typo + answer[i+1:]
                test_cases.append((test_input, f"자음/모음 오타({i}번째)"))
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
    synonyms = {
        '편이라': ['편', '편이다'],
        '남 말한다': ['남 나무란다'],
        '올챙이 시절': ['올챙이 적'],
        '개도 안 때린다': ['개도 안 건드린다'],
        '나무에서 떨어진다': ['나무에서 떨어질 때가 있다'],
        '도구를 탓하지 않는다': ['연장을 탓하지 않는다'],
        '개도 안 때린다': ['개도 안 건드린다'],
        '댓돌을 뚫는다': ['바위 뚫는다'],
    }
    words = answer.split()
    for i, word in enumerate(words):
        if word in synonyms:
            for synonym in synonyms[word]:
                new_words = words.copy()
                new_words[i] = synonym
                test_input = ' '.join(new_words)
                test_cases.append((test_input, f"동의어 교체({word}→{synonym})"))
    if len(answer) >= 2:
        for i in range(1, len(answer)):
            test_input = answer[:i] + "가" + answer[i:]
            test_cases.append((test_input, f"글자 추가({i}번째)"))
        for i in range(len(answer)):
            test_input = answer[:i] + answer[i+1:]
            test_cases.append((test_input, f"글자 제거({i}번째)"))
    if len(answer) >= 2:
        for i in range(len(answer) - 1):
            chars = list(answer)
            chars[i], chars[i+1] = chars[i+1], chars[i]
            test_input = ''.join(chars)
            test_cases.append((test_input, f"순서 바꾸기({i},{i+1})"))
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
    if " " in answer:
        words = answer.split()
        if len(words) >= 2:
            reversed_words = words[::-1]
            test_input = ' '.join(reversed_words)
            test_cases.append((test_input, "단어 순서 바꾸기"))
    return test_cases

def check_proverb(user_input, answers, threshold):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    answer_embeddings = model.encode(answers)
    user_embedding = model.encode([user_input])[0]
    similarities = util.cos_sim(user_embedding, answer_embeddings)[0]
    max_idx = similarities.argmax()
    max_score = similarities[max_idx].item()
    matched_answer = answers[max_idx]
    if max_score >= threshold:
        return True, matched_answer, max_score
    else:
        return False, matched_answer, max_score

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