# 프로젝트명

## 프로젝트 개요
이 프로젝트는 유사도 판별과 난이도 분석 기능을 제공하는 통합 시스템입니다.

## 전체 아키텍처
```
├── frontend/                    # 프론트엔드 (UI/UX)
├── services/                    # 백엔드 서비스들
│   ├── similarity/              # 유사도 판별 서비스
│   │   ├── modules/            # 유사도 분석 핵심 로직
│   │   ├── api/                # REST API 엔드포인트
│   │   ├── tests/              # 테스트 코드
│   │   └── docs/               # 문서화
│   └── difficulty/              # 난이도 판별 서비스
│       ├── modules/            # AI 모델 및 핵심 로직
│       ├── api/                # REST API 엔드포인트
│       ├── tests/              # 테스트 코드
│       └── docs/               # 문서화
└── database/                   # 데이터베이스 관리
```

## 팀 구성 및 담당 영역
- **프론트엔드 팀원**: 사용자 인터페이스 개발
- **유사도 서비스 팀원**: 유사도 분석 API 개발
- **난이도 서비스 팀원**: 난이도 판별 AI 시스템 개발
- **데이터베이스 팀원**: DB 설계 및 관리

## 시작하기

### 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt
```

### 각 서비스별 실행
각 폴더의 README.md를 참고하여 해당 서비스를 실행하세요.

## 개발 규칙
- 각 팀원은 담당 폴더에서만 작업
- 공통 의존성은 루트의 requirements.txt에 추가
- API 변경 시 문서화 필수
- 코드 리뷰 후 메인 브랜치 병합

## API 문서
- 유사도 API: `services/similarity/docs/` 참고
- 난이도 AI API: `services/difficulty/docs/` 참고

## 마이크로서비스 구조
각 서비스는 독립적으로 실행 가능하며, API Gateway를 통해 통합됩니다.

## 배포
TBD (배포 방법은 추후 문서화 예정)
