"""
🎯 속담 게임 - 속담 난이도 판별 기능 전용 설정 파일

이 시스템은 속담 미니게임의 핵심 기능입니다:
- 속담 절반을 주고 나머지 속담을 맞추는 게임
- AI가 속담의 난이도를 자동 판별하여 점수 차등 지급
- jhgan/ko-sroberta-multitask 모델만 사용 (다른 모델 사용 금지)

주요 기능:
1. 속담 텍스트 임베딩 생성
2. 속담 난이도 자동 분석 (1-5단계)
3. 난이도별 점수 배율 적용
4. MySQL 데이터베이스 연동 (proverb_game.proverb 테이블)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

class ProverbDifficultyConfig:
    """
    🎯 속담 게임 - 속담 난이도 판별 전용 설정 관리 클래스
    
    이 클래스는 속담 미니게임의 난이도 분석 기능만을 위한 설정을 관리합니다.
    - 속담 텍스트 분석 전용
    - jhgan/ko-sroberta-multitask 모델 전용
    - 속담 난이도별 점수 차등 지급 시스템
    """
    
    # ==================== 속담 게임 기본 설정 ====================
    PROJECT_NAME: str = "🎯 속담 게임 - 속담 난이도 판별 시스템"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "속담 절반을 주고 나머지를 맞추는 게임의 난이도 분석 AI"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # 현재 파일의 경로를 기준으로 프로젝트 루트 경로 설정
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # ==================== 속담 AI 모델 설정 ====================
    # 🚨 중요: jhgan/ko-sroberta-multitask 모델만 사용 (다른 모델 절대 금지!)
    MODEL_NAME: str = "jhgan/ko-sroberta-multitask"  # 한국어 속담 분석 전용 모델
    MODEL_CACHE_DIR: str = os.path.join(BASE_DIR, "app", "includes", "proverb_models", "cache")  # 속담 모델 캐시
    
    # GPU/CPU 자동 감지 (속담 분석 최적화)
    DEVICE: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    # 속담 텍스트 처리 설정
    MAX_SEQUENCE_LENGTH: int = 128  # 속담은 일반적으로 짧으므로 128로 설정
    BATCH_SIZE: int = 16 if DEVICE == "cuda" else 8  # 속담 배치 처리 크기
    
    # 배치 처리 설정
    BATCH_SIZE_ANALYSIS: int = 16   # 난이도 분석 배치 크기 (메모리 최적화)
    ENABLE_CACHING: bool = True     # 분석 결과 캐싱 활성화
    
    # ==================== 속담 게임 데이터베이스 설정 ====================
    # MySQL 속담 게임 전용 데이터베이스 연결 정보
    # 🚨 중요: proverb_game 데이터베이스, proverb 테이블 전용
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "0000")  # 데이터베이스 비밀번호
    DB_NAME: str = os.getenv("DB_NAME", "proverb_game")  # 속담 게임 전용 DB
    
    # 속담 테이블 설정
    PROVERB_TABLE: str = "proverb"  # 속담 원문 저장 테이블
    SCORE_TABLE: str = "user"  # 사용자 점수 저장 테이블
    
    # 데이터베이스 연결 URL 생성
    @property
    def DATABASE_URL(self) -> str:
        """MySQL 데이터베이스 연결 URL을 반환합니다."""
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # ==================== API 서버 설정 ====================
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8080"))  # 포트 8080 사용
    API_RELOAD: bool = DEBUG  # 개발 모드에서만 자동 재로드
    
    # ==================== 속담 난이도 판별 시스템 설정 ====================
    # 🎯 속담 게임 전용 난이도 레벨 정의 (3단계)
    # 속담의 복잡성, 인지도, 사용 빈도를 기반으로 3단계 분류
    PROVERB_DIFFICULTY_LEVELS: Dict[int, Dict] = {
        1: {
            "name": "🟢 쉬움", 
            "description": "일상적이고 잘 알려진 속담",
            "score": 1,
            "examples": ["가는 말이 고와야 오는 말이 곱다", "호랑이도 제 말 하면 온다"]
        },
        2: {
            "name": "🟡 보통", 
            "description": "어느 정도 알려진 일반적인 속담",
            "score": 2,
            "examples": ["밤말은 새가 듣고 낮말은 쥐가 듣는다", "백문이 불여일견"]
        },
        3: {
            "name": "🔴 어려움", 
            "description": "복잡하거나 잘 알려지지 않은 속담",
            "score": 3,
            "examples": ["가자니 태산이요, 돌아서자니 숭산이라", "금강산도 식후경"]
        }
    }
    
    # 속담 게임 점수 설정
    MIN_SCORE: int = 1      # 최소 점수 (쉬움)
    MAX_SCORE: int = 3      # 최대 점수 (어려움)
    BONUS_SCORE: int = 1    # 연속 정답 보너스
    
    # ==================== 로깅 설정 ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG")
    LOG_FILE: str = os.path.join(BASE_DIR, "logs", "difficulty_service.log")
    
    # ==================== 캐시 설정 ====================
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 캐시 유지 시간 (초)
    
    # ==================== 보안 설정 ====================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    def __init__(self):
        """
        설정 초기화 및 필요한 디렉토리 생성
        """
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self) -> None:
        """
        필요한 디렉토리들을 생성합니다.
        """
        directories = [
            self.MODEL_CACHE_DIR,  # 모델 캐시 디렉토리
            os.path.dirname(self.LOG_FILE),  # 로그 디렉토리
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_config(self) -> None:
        """
        설정값들의 유효성을 검사합니다.
        """
        # 필수 설정값 검사
        required_settings = {
            "MODEL_NAME": self.MODEL_NAME,
            "DB_HOST": self.DB_HOST,
            "DB_NAME": self.DB_NAME,
        }
        
        for setting_name, setting_value in required_settings.items():
            if not setting_value:
                raise ValueError(f"필수 설정값이 누락되었습니다: {setting_name}")
    
    def get_device_info(self) -> dict:
        """
        현재 사용 중인 디바이스 정보를 반환합니다.
        
        Returns:
            dict: 디바이스 정보 (디바이스 타입, GPU 이름, 메모리 정보 등)
        """
        device_info = {
            "device": self.DEVICE,
            "device_name": "CPU"
        }
        
        if self.DEVICE == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            device_info.update({
                "device_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB"
            })
        
        return device_info
    
    def print_config_summary(self) -> None:
        """
        🎯 속담 게임 난이도 판별 시스템 설정 요약 출력
        개발 및 디버깅 시 유용합니다.
        """
        print("=" * 70)
        print(f"🎯 {self.PROJECT_NAME} v{self.VERSION}")
        print(f"📝 {self.DESCRIPTION}")
        print("=" * 70)
        print(f"📍 실행 모드: {'개발 모드' if self.DEBUG else '운영 모드'}")
        print(f"🤖 AI 모델: {self.MODEL_NAME} (속담 분석 전용)")
        print(f"💻 디바이스: {self.DEVICE.upper()}")
        print(f"🗄️  속담 DB: {self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")
        print(f"📊 속담 테이블: {self.PROVERB_TABLE}")
        print(f"🌐 API 서버: {self.API_HOST}:{self.API_PORT}")
        print(f"📁 모델 캐시: {self.MODEL_CACHE_DIR}")
        print(f"🎮 난이도 레벨: {len(self.PROVERB_DIFFICULTY_LEVELS)}단계")
        print("=" * 70)
    
    def get_test_proverbs(self) -> List[str]:
        """
        🧪 테스트용 속담 목록을 반환합니다.
        각 난이도별 대표 속담들로 구성됩니다.
        
        Returns:
            List[str]: 테스트용 속담 리스트
        """
        test_proverbs = []
        for level_info in self.PROVERB_DIFFICULTY_LEVELS.values():
            test_proverbs.extend(level_info["examples"])
        return test_proverbs


# 전역 설정 인스턴스 생성
# 다른 모듈에서 'from config import proverb_config'로 사용할 수 있습니다.
proverb_config = ProverbDifficultyConfig()


if __name__ == "__main__":
    """
    🧪 이 파일을 직접 실행하면 속담 게임 설정 정보를 출력합니다.
    터미널에서 'python config.py' 명령으로 테스트할 수 있습니다.
    """
    # 속담 게임 설정 정보 출력
    proverb_config.print_config_summary()
    
    # 디바이스 정보 출력
    print("\n💻 디바이스 상세 정보:")
    device_info = proverb_config.get_device_info()
    for key, value in device_info.items():
        print(f"  - {key}: {value}")
    
    print(f"\n🎮 속담 난이도 레벨 설정:")
    for level, info in proverb_config.PROVERB_DIFFICULTY_LEVELS.items():
        print(f"  - 레벨 {level} {info['name']}: {info['score']}점")
        print(f"    📝 {info['description']}")
        print(f"    📚 예시: {', '.join(info['examples'])}")
        print()
    
    print(f"🧪 테스트용 속담 목록:")
    test_proverbs = proverb_config.get_test_proverbs()
    for i, proverb in enumerate(test_proverbs, 1):
        print(f"  {i}. {proverb}")
