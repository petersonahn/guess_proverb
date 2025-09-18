import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random
import time

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
    if length <= 2:
        return 0.5
    elif length == 3:
        return 0.55
    elif length == 4:
        return 0.6
    elif length == 5:
        return 0.65
    elif 6 <= length <= 7:
        return 0.7
    elif 8 <= length <= 10:
        return 0.75
    else:
        return 0.8

def check_proverb(user_input, answers, threshold):
    model = SentenceTransformer('jhgan/ko-sbert-nli')
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