#!/usr/bin/env python3
"""
프로젝트 폴더 구조 자동 생성 스크립트
실행 방법: python setup_project.py
"""

import os
import sys

def create_directory(path):
    """디렉토리 생성 (이미 존재하면 무시)"""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"✓ 디렉토리 생성: {path}")
    except Exception as e:
        print(f"✗ 디렉토리 생성 실패: {path} - {e}")
        return False
    return True

def create_file(filepath, content):
    """파일 생성"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 파일 생성: {filepath}")
    except Exception as e:
        print(f"✗ 파일 생성 실패: {filepath} - {e}")
        return False
    return True

def setup_project():
    """프로젝트 폴더 구조 생성"""
    
    print("🚀 프로젝트 폴더 구조 생성을 시작합니다...\n")
    
    # 디렉토리 구조 정의
    directories = [
        "frontend",
        "services",
        # 유사도 판별 서비스
        "services/similarity",
        "services/similarity/modules",
        "services/similarity/api",
        "services/similarity/tests",
        "services/similarity/docs",
        # 난이도 판별 서비스
        "services/difficulty",
        "services/difficulty/modules",
        "services/difficulty/api",
        "services/difficulty/tests",
        "services/difficulty/docs",
        # 데이터베이스
        "database",
    ]
    
    # 디렉토리 생성
    print("📁 디렉토리 생성 중...")
    for directory in directories:
        if not create_directory(directory):
            print(f"프로젝트 설정 중단: {directory} 생성 실패")
            return False
    
    print("\n📄 파일 생성 중...")
    
    # 파일 내용 정의
    files_content = {
        # 프론트엔드 README
        "frontend/README.md": """# Frontend

이 폴더는 **프론트엔드 팀원**이 작업합니다.

## 담당 업무
- 사용자 인터페이스 개발
- 사용자 경험(UX) 구현
- API 연동 및 데이터 표시

## 기술 스택
- React / Vue.js / Angular 등
- HTML, CSS, JavaScript
- 상태 관리 라이브러리

## 시작하기
```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```
""",

        # 유사도 판별 서비스 README
        "services/similarity/README.md": """# Similarity Service

이 폴더는 **유사도 판별 기능 백엔드 팀원**이 작업합니다.

## 담당 업무
- 텍스트/이미지 유사도 분석 API 개발
- 데이터 전처리 및 특징 추출
- 유사도 알고리즘 구현 및 최적화

## 주요 기능
- 문서 간 유사도 계산
- 이미지 유사도 분석
- 벡터 유사도 측정
- 실시간 유사도 판별

## 폴더 구조
- `modules/`: 유사도 분석 핵심 로직
- `api/`: REST API 엔드포인트
- `tests/`: 단위 테스트 및 통합 테스트
- `docs/`: API 문서 및 알고리즘 설명서

## 기술 스택
- FastAPI / Flask
- NumPy, Pandas
- scikit-learn, Sentence Transformers
- Redis (캐싱)
""",

        # 난이도 판별 서비스 README
        "services/difficulty/README.md": """# Difficulty Service

이 폴더는 **난이도 판별 기능 백엔드 팀원**이 작업합니다.

## 담당 업무
- AI 모델을 활용한 난이도 분석 시스템 개발
- 머신러닝 모델 학습 및 배포
- 난이도 예측 API 구현

## 주요 기능
- 텍스트 난이도 자동 분석
- 문제 난이도 등급 분류
- 학습 수준별 콘텐츠 추천
- 실시간 난이도 예측

## 폴더 구조
- `modules/`: AI 모델 및 핵심 로직
- `api/`: REST API 엔드포인트
- `tests/`: 단위 테스트 및 통합 테스트
- `docs/`: API 문서 및 모델 설명서

## 기술 스택
- FastAPI
- TensorFlow / PyTorch
- Transformers (Hugging Face)
- MLflow (모델 관리)
""",

        # 데이터베이스 README
        "database/README.md": """# Database

이 폴더는 **데이터베이스 팀원**이 작업합니다.

## 담당 업무
- 데이터베이스 스키마 설계
- 데이터 모델링 및 최적화
- 마이그레이션 스크립트 관리
- 백업 및 복구 전략 수립

## 주요 업무
- ERD 설계 및 테이블 구조 최적화
- 인덱스 및 성능 튜닝
- 데이터 무결성 보장
- 쿼리 최적화

## 기술 스택
- PostgreSQL / MySQL / MongoDB
- Redis (캐싱)
- 데이터베이스 마이그레이션 도구
- 백업 및 모니터링 도구
""",

        # 메인 프로젝트 README
        "README.md": """# 프로젝트명

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
""",

        # .gitignore
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Database
*.db
*.sqlite
*.sqlite3

# Node.js (for frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build outputs
dist/
build/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Model files
*.pkl
*.joblib
*.h5
*.pb
models/checkpoints/

# Data files
*.csv
*.json
data/raw/
data/processed/

# Service specific
**/temp/
**/cache/
""",

        # requirements.txt
        "requirements.txt": """# Web Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# 데이터베이스
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.1

# 데이터 처리
pandas==2.1.4
numpy==1.25.2

# 유사도 서비스용
scikit-learn==1.3.2
sentence-transformers==2.2.2
nltk==3.8.1
faiss-cpu==1.7.4

# 난이도 서비스용 (AI/ML)
torch==2.1.1
transformers==4.36.0
tensorflow==2.15.0

# 캐싱 및 큐
redis==5.0.1
celery==5.3.4

# HTTP 클라이언트
httpx==0.25.2
requests==2.31.0

# 개발 도구
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
mypy==1.7.1

# 모니터링
prometheus-client==0.19.0

# 기타 유틸리티
python-dotenv==1.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
"""
    }
    
    # 파일 생성
    for filepath, content in files_content.items():
        if not create_file(filepath, content):
            print(f"프로젝트 설정 중단: {filepath} 생성 실패")
            return False
    
    print(f"\n🎉 프로젝트 구조가 성공적으로 생성되었습니다!")
    print(f"📂 현재 폴더에 생성완료")
    print("\n📋 생성된 구조:")
    print_directory_tree(".")
    
    print("\n📝 다음 단계:")
    print("1. python -m venv venv")
    print("2. source venv/bin/activate  # Windows: venv\\Scripts\\activate")
    print("3. pip install -r requirements.txt")
    print("4. 각 팀원별로 담당 서비스에서 개발 시작")

def print_directory_tree(root_path, prefix="", is_last=True):
    """디렉토리 트리 출력"""
    if not os.path.exists(root_path):
        return
        
    items = sorted(os.listdir(root_path))
    dirs = [item for item in items if os.path.isdir(os.path.join(root_path, item))]
    files = [item for item in items if os.path.isfile(os.path.join(root_path, item))]
    
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last_item = (i == len(all_items) - 1)
        current_prefix = "└── " if is_last_item else "├── "
        print(f"{prefix}{current_prefix}{item}")
        
        if item in dirs:
            extension = "    " if is_last_item else "│   "
            print_directory_tree(
                os.path.join(root_path, item), 
                prefix + extension, 
                is_last_item
            )

if __name__ == "__main__":
    try:
        setup_project()
    except KeyboardInterrupt:
        print("\n\n❌ 설정이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)