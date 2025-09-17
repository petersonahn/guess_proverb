# Similarity Service

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
