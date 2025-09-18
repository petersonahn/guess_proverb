from sentence_transformers import SentenceTransformer, util
import mysql.connector
import re

# 테스트할 언어 유사도 모델 리스트
MODEL_NAMES = [
    'jhgan/ko-sbert-nli',
    'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
]

def get_proverbs_from_db():
    conn = mysql.connector.connect(
        host='localhost', user='root', password='1234', database='proverb_game'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer, hint FROM proverb")
    proverbs = cursor.fetchall()
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

def generate_user_inputs(answer):
    user_inputs = []
    # 1. 완전 정답
    user_inputs.append((answer, True))
    # 2. 정답+특수문자
    user_inputs.append((answer + '!', True))
    # 3. 공백 제거
    user_inputs.append((answer.replace(' ', ''), True))
    # 4. 정답 반복
    user_inputs.append((answer + ' ' + answer, True))
    # 5. 일부 단어만(앞/뒤/중간)
    words = answer.split()
    if len(words) > 1:
        user_inputs.append((' '.join(words[:-1]), False))
        user_inputs.append((' '.join(words[1:]), False))
    # 6. 앞부분만/뒷부분만
    if len(answer) > 2:
        user_inputs.append((answer[:len(answer)//2], False))
        user_inputs.append((answer[-(len(answer)//2):], False))
    # 7. 오타(한 글자 바꾸기)
    if len(answer) > 1:
        typo = answer[:-1] + chr((ord(answer[-1]) + 1) % 0xAC00 + 0xAC00)
        user_inputs.append((typo, False))
    # 8. 조사가 다른 경우 (예: '가', '이', '를', '을', '은', '는', '도', '와', '과' 등)
    for josa in ['가', '이', '를', '을', '은', '는', '도', '와', '과']:
        if not answer.endswith(josa):
            user_inputs.append((answer + josa, True))
    # 9. 어미가 다른 경우 (예: '다', '요', '임', '임다', '임요')
    if not answer.endswith('다'):
        user_inputs.append((answer + '다', True))
    user_inputs.append((answer + '요', True))
    user_inputs.append((answer + '임', True))
    user_inputs.append((answer + '임다', True))
    user_inputs.append((answer + '임요', True))
    # 10. 띄어쓰기 추가/제거
    user_inputs.append((re.sub(r'(.)', r'\1 ', answer).strip(), True))
    # 11. 완전 오답
    user_inputs.append(('전혀다른답', False))
    return user_inputs

def test_model_accuracy(model_name, proverbs):
    model = SentenceTransformer(model_name)
    total = 0
    correct = 0
    detailed = []
    for pid, question, answer, hint in proverbs:
        threshold = get_threshold_by_length(answer)
        user_inputs = generate_user_inputs(answer)
        for user_input, expected in user_inputs:
            answer_embedding = model.encode([answer])
            user_embedding = model.encode([user_input])[0]
            similarity = util.cos_sim(user_embedding, answer_embedding)[0][0].item()
            is_correct = similarity >= threshold
            total += 1
            if is_correct == expected:
                correct += 1
            detailed.append({
                'question': question,
                'answer': answer,
                'user_input': user_input,
                'expected': expected,
                'similarity': similarity,
                'model_judge': is_correct
            })
    accuracy = correct / total if total > 0 else 0
    return accuracy, detailed

def main():
    proverbs = get_proverbs_from_db()
    with open('services/similarity/similarity_test2.txt', 'w', encoding='utf-8') as f:
        for model_name in MODEL_NAMES:
            f.write(f'=== 모델: {model_name} ===\n')
            accuracy, details = test_model_accuracy(model_name, proverbs)
            f.write(f'정확도: {accuracy:.2%}\n')
            for d in details:
                f.write(f"Q: {d['question']} | 정답: {d['answer']} | 입력: {d['user_input']} | 기대: {d['expected']} | 유사도: {d['similarity']:.2f} | 모델판정: {d['model_judge']}\n")
            f.write('\n')
    print('테스트 완료! 결과는 similarity_test2.txt에서 확인하세요.')

if __name__ == '__main__':
    main()
