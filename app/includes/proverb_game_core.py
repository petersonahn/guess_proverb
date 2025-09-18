import mysql.connector
from sentence_transformers import SentenceTransformer, util
import random

MODEL = None
PROVERBS = None

# 모델 로딩 (최초 1회)
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer('jhgan/ko-sbert-nli')
    return MODEL

# DB에서 속담 전체 불러오기 (최초 1회 캐싱)
def get_proverbs():
    global PROVERBS
    if PROVERBS is None:
        conn = mysql.connector.connect(
            host='localhost', user='root', password='1234', database='proverb_game'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, answer, hint FROM proverb")
        PROVERBS = cursor.fetchall()
        cursor.close()
        conn.close()
    return PROVERBS

def get_random_proverb():
    proverbs = get_proverbs()
    if not proverbs:
        return None
    pid, question, answer, hint = random.choice(proverbs)
    return {"id": pid, "question": question, "hint": hint}

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

def check_proverb_answer(proverb_id, user_answer):
    proverbs = get_proverbs()
    proverb = next((p for p in proverbs if p[0] == proverb_id), None)
    if not proverb:
        return {"result": "not_found"}
    _, _, answer, hint = proverb
    threshold = get_threshold_by_length(answer)
    model = get_model()
    answers = [p[2] for p in proverbs]
    answer_embeddings = model.encode(answers)
    user_embedding = model.encode([user_answer])[0]
    similarities = util.cos_sim(user_embedding, answer_embeddings)[0]
    max_idx = similarities.argmax()
    max_score = similarities[max_idx].item()
    matched_answer = answers[max_idx]
    is_correct = max_score >= threshold and matched_answer == answer
    return {
        "is_correct": is_correct,
        "matched": matched_answer,
        "score": max_score,
        "answer": answer if not is_correct else None,
        "hint": hint if not is_correct else None
    }
