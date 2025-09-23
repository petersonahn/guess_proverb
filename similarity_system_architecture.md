# 🧠 AI 학습 기반 속담 유사도 검사 시스템 구조

## 📋 시스템 개요

기존의 하드코딩된 방식을 완전히 뜯어고쳐서 **AI 모델이 proverb 테이블의 실제 데이터를 학습**하여 지능적으로 예외사항을 감지하고 처리하는 시스템으로 구축했습니다.

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI 학습 기반 Similarity 시스템                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  데이터 학습 단계  │    │   검사 실행 단계  │    │  백업 시스템   │ │
│  │                 │    │                 │    │              │ │
│  │ ProverbDataLearner │    │ IntelligentProverbChecker │    │ EnhancedChecker │ │
│  │                 │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           ▼                       ▼                       │     │
│  ┌─────────────────┐    ┌─────────────────┐              │     │
│  │  패턴 학습 모듈   │    │  지능적 검사 모듈 │              │     │
│  │                 │    │                 │              │     │
│  │ • 동물 단어 추출  │    │ • 동물 혼동 감지  │              │     │
│  │ • 부정 표현 학습  │    │ • 의미 반전 감지  │              │     │
│  │ • 반의어 관계 학습 │    │ • 핵심 단어 검사  │              │     │
│  │ • 문맥 그룹 생성  │    │ • 종합 점수 계산  │              │     │
│  │ • 중요도 계산    │    │                 │              │     │
│  └─────────────────┘    └─────────────────┘              │     │
│           │                       │                       │     │
│           ▼                       ▼                       │     │
│  ┌─────────────────┐    ┌─────────────────┐              │     │
│  │  학습된 패턴 저장  │    │  최종 유사도 점수 │              │     │
│  │                 │    │                 │              │     │
│  │ • learned_patterns │    │ • base_similarity │              │     │
│  │ • animal_groups   │    │ • AI_penalties  │              │     │
│  │ • opposite_pairs  │    │ • final_score   │              │     │
│  │ • critical_words  │    │                 │              │     │
│  └─────────────────┘    └─────────────────┘              │     │
│                                                           │     │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 동작 원리

### 1단계: 데이터 학습 (ProverbDataLearner)

```python
# proverb 테이블에서 84개 속담 데이터 로드
proverb_data = load_proverb_data()

# 각종 패턴 자동 학습
animals = extract_animals_from_data()        # 40개 동물 단어 추출
negations = extract_negation_patterns()      # 11개 부정 표현 추출
opposites = learn_semantic_opposites()       # BERT 기반 반의어 학습
contexts = learn_context_groups()           # TF-IDF 클러스터링
critical = learn_critical_words()           # 중요도 기반 핵심 단어
```

**학습 결과 예시:**
```python
learned_patterns = {
    'animals': {'개', '게', '소', '말', '개구리', '올챙이', ...},
    'animal_groups': {
        '갑각류': ['게', '새우', '꼴뚜기'],
        '포유류': ['개', '소', '말', '고양이', '호랑이', ...],
        '조류': ['새', '뱁새', '황새']
    },
    'opposite_pairs': {
        '있다': ['없다', '없어'],
        '좋다': ['나쁘다', '나빠'],
        '알다': ['모르다', '모른다']
    },
    'critical_words': {
        '개구리': 0.30,    # 70% 감점
        '올챙이': 0.30,    # 70% 감점
        '쳇바퀴': 0.20,    # 80% 감점
        '우물': 0.50       # 50% 감점
    }
}
```

### 2단계: 지능적 검사 (IntelligentProverbChecker)

```python
def comprehensive_check(original, user_input, base_similarity):
    final_score = base_similarity
    
    # 1. 동물 혼동 감지
    animal_penalty = detect_animal_confusion(original, user_input)
    # "게" → "개" 감지시 0.01 (99% 감점)
    
    # 2. 의미적 반전 감지  
    opposite_penalty = detect_semantic_opposite(original, user_input)
    # "고와야" → "나쁘면" 감지시 0.01 (99% 감점)
    
    # 3. 핵심 단어 누락 감지
    critical_penalty = detect_critical_word_missing(original, user_input)
    # 학습된 중요도에 따라 차등 패널티
    
    final_score *= animal_penalty * opposite_penalty * critical_penalty
    return min(final_score, 1.0)
```

## 🎯 핵심 검사 로직

### 1. 동물 혼동 감지
```python
def detect_animal_confusion(original, user_input):
    # 원본: "마파람에 게눈 감추듯"
    # 입력: "마파람에 개눈 감추듯"
    
    orig_animals = ['게']      # 갑각류
    user_animals = ['개']      # 포유류
    
    # 다른 그룹의 동물로 바뀜 → 99% 감점
    if orig_group != user_group:
        return 0.01
```

### 2. 의미적 반전 감지
```python
def detect_semantic_opposite(original, user_input):
    # 원본: "가는 말이 고와야"
    # 입력: "가는 말이 나쁘면"
    
    # BERT 모델로 의미적 유사도 계산
    similarity = bert_model.similarity("고와야", "나쁘면")
    
    # 유사도가 매우 낮으면 반대 의미로 판단
    if similarity < 0.3:
        return 0.01  # 99% 감점
```

### 3. 핵심 단어 누락 감지
```python
def detect_critical_word_missing(original, user_input):
    # 학습된 중요도 기반 패널티 적용
    for word, penalty_factor in critical_words.items():
        if word in original and word not in user_input:
            penalty *= penalty_factor
    
    # 예: "개구리" 누락시 0.30 (70% 감점)
    # 예: "우물" 누락시 0.50 (50% 감점)
```

## 🔧 시스템 통합

### 메인 검사 함수 (similarity_check.py)
```python
def check_proverb(user_input, answers, threshold):
    # 1순위: AI 학습 시스템 사용
    if AI_LEARNING_AVAILABLE:
        checker = IntelligentProverbChecker()
        return checker.comprehensive_check(...)
    
    # 2순위: 향상된 시스템 사용 (백업)
    elif ENHANCED_CHECKER_AVAILABLE:
        checker = EnhancedProverbSimilarityChecker()
        return checker.check_proverb(...)
    
    # 3순위: 기본 BERT 시스템 사용 (최종 백업)
    else:
        return basic_bert_check(...)
```

## 📊 성능 비교

| 구분 | 기존 하드코딩 방식 | AI 학습 방식 |
|------|-------------------|--------------|
| **동물 혼동** | `ANIMAL_PAIRS = [('게', '개')]` | 자동 학습으로 40개 동물 분류 |
| **의미 반전** | `CRITICAL_TYPOS = {'없다': ['있다']}` | BERT 기반 자동 감지 |
| **핵심 단어** | `CRITICAL_PROVERB_WORDS = {...}` | 중요도 기반 동적 계산 |
| **확장성** | 새 속담마다 하드코딩 필요 | 자동 학습으로 무한 확장 |
| **유지보수** | 수동으로 패턴 추가/수정 | 데이터 기반 자동 개선 |

## 🚀 주요 장점

### 1. **완전 자동화**
- proverb 테이블 데이터를 기반으로 자동 학습
- 새로운 속담 추가시 자동으로 패턴 학습
- 하드코딩 없이 지능적 처리

### 2. **정확한 감지**
- "게" → "개" 동물 혼동: 99% 감점
- "고와야" → "나쁘면" 의미 반전: 99% 감점  
- 핵심 단어 누락: 중요도에 따라 차등 감점

### 3. **확장성**
- 84개 속담 → 무제한 확장 가능
- 새로운 예외사항 자동 학습
- 데이터 기반 지속적 개선

### 4. **안정성**
- 3단계 백업 시스템 (AI → Enhanced → Basic)
- 오류 발생시 자동으로 하위 시스템 사용
- 기존 코드와 완전 호환

## 💡 사용 예시

```python
# 기존 코드 그대로 사용
from services.similarity.similarity_check import check_proverb

# AI가 자동으로 모든 예외사항을 감지
is_correct, matched, score = check_proverb(
    "마파람에 개눈 감추듯",  # 사용자 입력 (게→개 혼동)
    ["마파람에 게눈 감추듯"], # 정답
    0.5  # 임계값
)

# 결과: is_correct = False, score = 0.01 (99% 감점)
```

이제 하드코딩 없이도 AI가 proverb 테이블의 모든 예외사항을 자동으로 학습하고 처리할 수 있습니다! 🎉
