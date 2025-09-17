# 🎯 속담 게임 - 속담 난이도 판별 AI 시스템

## 📖 프로젝트 개요

이 시스템은 **속담 미니게임**의 핵심 기능을 담당합니다:

- **🎮 게임 방식**: 속담 절반을 주고 나머지 속담을 맞추는 게임
- **🤖 AI 기능**: 속담의 난이도를 자동 판별하여 점수 차등 지급
- **📊 점수 시스템**: 난이도별 점수 배율 적용 (쉬움 1x ~ 최고난이도 3x)
- **🎯 전용 모델**: jhgan/ko-sroberta-multitask 모델만 사용

## 🚨 중요 사항

- **속담 게임 전용 시스템**입니다 (범용 텍스트 분석 아님)
- **jhgan/ko-sroberta-multitask 모델만 사용** (다른 모델 절대 금지)
- MySQL `proverb_game` 데이터베이스의 `proverb` 테이블과 연동
- 사용자별 속담 정답 점수 기록 및 관리

## 🏗️ 시스템 구조

```
services/difficulty/
├── modules/                    # 핵심 모듈
│   ├── requirements.txt       # 필수 패키지 목록
│   ├── config.py             # 속담 게임 전용 설정
│   └── model_test.py         # AI 모델 테스트 스크립트
├── api/                       # REST API 엔드포인트
├── tests/                     # 단위 테스트
├── docs/                      # 문서화
└── README.md                 # 이 파일
```

## 🎮 속담 난이도 시스템

### 난이도 레벨 (5단계)

1. **🟢 쉬움 (1x)**: 일상적이고 잘 알려진 속담
   - 예시: "가는 말이 고와야 오는 말이 곱다", "호랑이도 제 말 하면 온다"
   - 기본 점수: 100점

2. **🔵 보통 (1.5x)**: 어느 정도 알려진 일반적인 속담
   - 예시: "백문이 불여일견", "천리 길도 한 걸음부터"
   - 기본 점수: 150점

3. **🟡 어려움 (2x)**: 교육받은 사람들이 아는 속담
   - 예시: "금강산도 식후경", "낮말은 새가 듣고 밤말은 쥐가 듣는다"
   - 기본 점수: 200점

4. **🟠 매우 어려움 (2.5x)**: 전문적 지식이 필요한 속담
   - 예시: "등잔 밑이 어둡다", "개천에서 용 난다"
   - 기본 점수: 250점

5. **🔴 최고 난이도 (3x)**: 고전 문학이나 역사적 배경 지식 필요
   - 예시: "하늘이 무너져도 솟아날 구멍이 있다", "닭 쫓던 개 지붕 쳐다본다"
   - 기본 점수: 300점

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화 (conda 또는 venv)
conda activate your_env
# 또는
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 필수 패키지 설치
cd services/difficulty/modules
pip install -r requirements.txt
```

### 2. 시스템 설정 확인

```bash
# 속담 게임 설정 정보 확인
python config.py
```

### 3. AI 모델 테스트

```bash
# 속담 난이도 판별 AI 모델 테스트 실행
python model_test.py
```

## 🧪 테스트 과정

`model_test.py` 실행 시 다음 테스트가 수행됩니다:

1. **🔍 시스템 요구사항 검사**
   - Python 버전, PyTorch 설치 확인
   - GPU/CPU 환경 자동 감지
   - 메모리 사용량 체크

2. **🤖 AI 모델 다운로드 및 로딩**
   - jhgan/ko-sroberta-multitask 모델 자동 다운로드
   - SentenceTransformer 및 Transformers 라이브러리 로딩
   - 모델 캐시 디렉토리 생성

3. **📝 속담 임베딩 생성 테스트**
   - 난이도별 대표 속담으로 임베딩 생성
   - 처리 속도 및 성능 측정

4. **🔍 속담 유사도 분석**
   - 같은 난이도 속담 간 유사도 계산
   - 다른 난이도 속담 간 차이점 분석

5. **🎯 난이도 분석 시뮬레이션**
   - 실제 게임 상황 시뮬레이션
   - 점수 계산 및 배율 적용 테스트

## 📊 데이터베이스 연동

### MySQL 설정

```sql
-- 데이터베이스 생성
CREATE DATABASE proverb_game;

-- 속담 테이블 (예시)
USE proverb_game;
CREATE TABLE proverb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    content VARCHAR(255) NOT NULL,
    difficulty_level INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 점수 테이블 (예시)
CREATE TABLE user_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    proverb_id INT NOT NULL,
    score INT NOT NULL,
    difficulty_level INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (proverb_id) REFERENCES proverb(id)
);
```

### 연결 설정

`modules/config.py`에서 데이터베이스 연결 정보를 설정하세요:

```python
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = "your_password"
DB_NAME = "proverb_game"
```

## 🔧 문제 해결

### 일반적인 문제

1. **모델 다운로드 실패**
   ```bash
   # 네트워크 연결 확인
   ping huggingface.co
   
   # 캐시 디렉토리 권한 확인
   ls -la services/difficulty/proverb_models/
   ```

2. **메모리 부족**
   ```python
   # config.py에서 배치 크기 조정
   BATCH_SIZE = 4  # GPU 메모리 부족 시
   ```

3. **CUDA 오류**
   ```bash
   # CPU 모드로 강제 실행
   export CUDA_VISIBLE_DEVICES=""
   python model_test.py
   ```

### 에러별 해결 방법

- **Connection Error**: 인터넷 연결 및 방화벽 설정 확인
- **Permission Error**: 관리자 권한으로 실행
- **Memory Error**: 다른 프로그램 종료 후 재시도
- **CUDA Out of Memory**: CPU 모드로 전환

## 📚 추가 정보

### 모델 정보
- **모델명**: jhgan/ko-sroberta-multitask
- **용도**: 한국어 문장 임베딩 생성
- **특징**: 다중 작업 학습으로 속담 분석에 최적화

### 성능 최적화
- GPU 사용 시 속담 분석 속도 10배 향상
- 배치 처리로 대량 속담 동시 분석 가능
- 캐시 시스템으로 재사용 속담 빠른 처리

### API 개발 예정
- 속담 난이도 분석 API
- 실시간 점수 계산 API
- 게임 상태 관리 API

## 🤝 기여 방법

1. 속담 데이터 추가
2. 난이도 분류 알고리즘 개선
3. 성능 최적화
4. 테스트 케이스 추가

## 📞 지원

문제 발생 시:
1. `model_test.py` 실행 결과 확인
2. 에러 로그 수집
3. 시스템 환경 정보 제공

---

**🎯 속담 게임의 핵심 AI 기능을 담당하는 전용 시스템입니다!**