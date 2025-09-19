# 🎯 속담 게임 - AI 난이도 분석 시스템

## 📋 프로젝트 개요
한국 속담의 난이도를 AI 모델로 자동 분석하고, 게임에서 사용할 점수를 부여하는 FastAPI 기반 백엔드 시스템입니다.

### 🎮 주요 기능
- **AI 난이도 분석**: jhgan/ko-sroberta-multitask 모델을 사용한 속담 난이도 자동 분석
- **3단계 점수 시스템**: 1점(쉬움), 2점(보통), 3점(어려움)
- **고성능 캐싱**: 메모리 기반 분석 결과 캐싱으로 빠른 응답
- **배치 처리**: 전체 속담 사전 분석 및 백그라운드 처리
- **RESTful API**: FastAPI 기반의 완전한 API 서버

## 🏗️ 시스템 아키텍처
```
├── app/                        # FastAPI 애플리케이션
│   ├── main.py                # 서버 메인 파일
│   ├── core/                  # 핵심 설정
│   │   └── config.py         # 설정 관리
│   ├── includes/              # 핵심 모듈
│   │   ├── analyzer.py       # AI 난이도 분석기
│   │   ├── dbconn.py        # 데이터베이스 연결
│   │   └── utils.py         # 유틸리티 함수
│   ├── routers/               # API 라우터
│   │   ├── proverbs.py      # 속담 조회 API
│   │   ├── analysis.py      # 난이도 분석 API
│   │   └── game.py          # 게임 API
│   └── schemas/               # Pydantic 모델
│       └── proverb.py       # API 스키마 정의
├── templates/                 # HTML 템플릿
├── static/                    # 정적 파일
├── requirements.txt           # Python 패키지
└── database/                  # 데이터베이스 관련
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 활성화
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 의존성 설치 (이미 설치됨)
pip install -r requirements.txt
```

### 2. 데이터베이스 설정
- **MySQL 서버**: localhost:3306
- **데이터베이스**: proverb_game
- **사용자**: root / 비밀번호: 0000
- **테이블**: proverb (id, question, answer, hint)

### 3. 서버 실행
```bash
# 방법 1: 자동 실행 스크립트 (권장)
python start_server.py

# 방법 2: uvicorn 직접 실행
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. 서버 확인
- **홈페이지**: http://localhost:8001
- **API 문서**: http://localhost:8001/docs
- **헬스 체크**: http://localhost:8001/health

## 📚 API 사용법

### 🔍 속담 조회
```bash
# 속담 목록 조회
GET /api/v1/proverbs?page=1&limit=10

# 개별 속담 조회
GET /api/v1/proverbs/1

# 랜덤 속담 조회
GET /api/v1/proverbs/random?difficulty=1

# 속담 검색
GET /api/v1/proverbs/search?q=가는
```

### 🤖 난이도 분석
```bash
# 개별 분석
POST /api/v1/analysis/single
{
  "proverb_id": 1,
  "force_refresh": false
}

# 배치 분석
POST /api/v1/analysis/batch
{
  "proverb_ids": [1, 2, 3],
  "force_refresh": false
}

# 전체 사전 분석 (백그라운드)
POST /api/v1/analysis/preload

# 캐시 상태 조회
GET /api/v1/analysis/cache
```

### 🎮 게임 API
```bash
# 문제 출제
GET /api/v1/game/question?difficulty=1

# 정답 확인
POST /api/v1/game/answer
{
  "proverb_id": 1,
  "user_answer": "오는 말이 곱다"
}

# 게임 통계
GET /api/v1/game/stats
```

## 🧪 테스트

### 구성 테스트
```bash
# FastAPI 서버 구성 확인
python quick_server_test.py
```

### API 테스트
```bash
# 전체 API 엔드포인트 테스트
python test_api.py
```

### 개별 모듈 테스트
```bash
# 전체 시스템 통합 테스트
python test_system.py

# 점수 확인
python check_scores.py --id 1           # 개별 속담
python check_scores.py --batch 5        # 5개 배치
python check_scores.py                  # 전체 속담
```

## 🔧 기술 스택

### Backend
- **FastAPI**: 고성능 웹 프레임워크
- **Pydantic**: 데이터 검증 및 직렬화
- **Uvicorn**: ASGI 서버

### AI/ML
- **jhgan/ko-sroberta-multitask**: 한국어 문장 임베딩 모델
- **Transformers**: Hugging Face 모델 로딩
- **PyTorch**: 딥러닝 프레임워크

### Database
- **MySQL**: 속담 데이터 저장
- **SQLAlchemy**: ORM
- **mysql-connector-python**: MySQL 드라이버

### DevOps
- **Docker**: 컨테이너화 (선택사항)
- **Conda**: 가상환경 관리
- **pytest**: 테스트 프레임워크

## 📊 성능 특징

- **캐싱**: 메모리 기반 분석 결과 캐싱으로 **0.001초** 응답
- **배치 처리**: 16개씩 묶어서 처리하여 메모리 효율성 확보
- **비동기 처리**: FastAPI의 비동기 기능으로 동시 요청 처리
- **백그라운드 작업**: 전체 속담 사전 분석을 백그라운드에서 실행

## 🎯 사용 시나리오

### 게임 개발자
1. **문제 출제**: `/api/v1/game/question`으로 난이도별 문제 조회
2. **정답 확인**: `/api/v1/game/answer`로 사용자 답안 검증
3. **점수 계산**: 난이도에 따른 자동 점수 부여 (1-3점)

### 데이터 분석가
1. **통계 조회**: `/api/v1/game/stats`로 전체 난이도 분포 확인
2. **배치 분석**: `/api/v1/analysis/batch`로 대량 속담 분석
3. **결과 내보내기**: 분석 결과를 JSON/CSV로 내보내기

## 🛠️ 개발 가이드

### 새로운 API 추가
1. `app/schemas/proverb.py`에 Pydantic 모델 정의
2. `app/routers/`에 라우터 파일 생성
3. `app/main.py`에 라우터 등록

### 설정 변경
- `app/core/config.py`: 모델 경로, DB 정보 등
- 환경변수 또는 파일 직접 수정

### 디버깅
- 로그 레벨: `logging.INFO`
- 개발 모드: `--reload` 플래그 사용
- API 문서: `/docs`에서 실시간 테스트
