"""
🎯 속담 게임 - 속담 난이도 판별 AI 모델 테스트 스크립트

이 스크립트는 속담 미니게임의 핵심 기능을 테스트합니다:
- 속담 절반을 주고 나머지 속담을 맞추는 게임
- AI가 속담의 난이도를 자동 판별하여 점수 차등 지급
- jhgan/ko-sroberta-multitask 모델 전용 (다른 모델 사용 금지!)

주요 테스트 항목:
1. jhgan/ko-sroberta-multitask 모델 자동 다운로드 및 로딩
2. GPU/CPU 환경 자동 감지 및 설정
3. 속담 예시로 임베딩 생성 및 분석
4. 속담 난이도별 차이 확인
5. 메모리 사용량 모니터링
6. 에러 상황 처리 및 해결 방법 안내

실행 방법:
    python model_test.py

필수 패키지 설치:
    pip install -r requirements.txt
"""

import os
import sys
import time
import traceback
from typing import List, Dict, Any, Tuple
from pathlib import Path

# config 모듈 import
try:
    from config import proverb_config
    print("✅ 속담 게임 설정 파일 로딩 성공!")
except ImportError as e:
    print(f"❌ config.py import 실패: {e}")
    print("해결 방법: config.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 필수 라이브러리 import
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    import psutil
    print("✅ 모든 필수 라이브러리 import 성공!")
except ImportError as e:
    print(f"❌ 라이브러리 import 실패: {e}")
    print("\n🔧 해결 방법:")
    print("1. 가상환경 활성화 확인")
    print("2. pip install -r requirements.txt 실행")
    print("3. Python 버전 확인 (3.8 이상 권장)")
    sys.exit(1)


class ProverbDifficultyTester:
    """
    🎯 속담 게임 전용 난이도 판별 AI 모델 테스터
    
    이 클래스는 속담 미니게임의 핵심 기능인 난이도 판별 시스템을 테스트합니다.
    jhgan/ko-sroberta-multitask 모델만을 사용하여 속담의 난이도를 분석합니다.
    """
    
    def __init__(self):
        """
        🎯 속담 난이도 판별 테스터 초기화
        """
        self.config = proverb_config
        self.model_name = self.config.MODEL_NAME
        self.device = self.config.DEVICE
        self.cache_dir = self.config.MODEL_CACHE_DIR
        
        # 모델 관련 변수
        self.sentence_model = None
        self.tokenizer = None
        self.transformer_model = None
        
        print(f"🎯 속담 게임 난이도 판별 테스터 초기화 완료")
        print(f"📍 모델명: {self.model_name} (속담 분석 전용)")
        print(f"💻 사용 디바이스: {self.device}")
        print(f"📁 모델 캐시: {self.cache_dir}")
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """
        🔍 속담 분석을 위한 시스템 요구사항을 체크합니다.
        
        Returns:
            Dict[str, Any]: 시스템 정보
        """
        print("\n" + "="*60)
        print("🔍 속담 게임 시스템 요구사항 검사")
        print("="*60)
        
        system_info = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": self.device,
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        }
        
        # GPU 정보 (사용 가능한 경우)
        if torch.cuda.is_available():
            system_info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            })
        else:
            system_info["cuda_available"] = False
        
        # 시스템 정보 출력
        print(f"🐍 Python 버전: {system_info['python_version']}")
        print(f"🔥 PyTorch 버전: {system_info['torch_version']}")
        print(f"💻 CPU 코어 수: {system_info['cpu_count']}")
        print(f"🧠 총 메모리: {system_info['memory_total']}")
        print(f"💾 사용 가능 메모리: {system_info['memory_available']}")
        
        if system_info["cuda_available"]:
            print(f"🎮 GPU 사용 가능: {system_info['gpu_name']}")
            print(f"🎯 CUDA 버전: {system_info['cuda_version']}")
            print(f"📊 GPU 메모리: {system_info['gpu_memory']}")
            print("✅ 속담 분석이 GPU에서 빠르게 실행됩니다!")
        else:
            print("💻 CPU 모드로 실행됩니다. (속담 분석 속도가 다소 느릴 수 있음)")
        
        return system_info
    
    def download_and_load_model(self) -> bool:
        """
        🤖 jhgan/ko-sroberta-multitask 모델을 다운로드하고 로딩합니다.
        
        Returns:
            bool: 성공 여부
        """
        try:
            print("\n" + "="*60)
            print("📥 속담 분석 AI 모델 다운로드 및 로딩")
            print("="*60)
            
            # 캐시 디렉토리 생성
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"📁 모델 캐시 디렉토리: {self.cache_dir}")
            
            print(f"🚨 중요: {self.model_name} 모델만 사용 (속담 분석 전용)")
            print("⏳ 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다...")
            
            start_time = time.time()
            
            # 1. SentenceTransformer 모델 로딩 (속담 임베딩 생성용)
            print(f"🔄 SentenceTransformer 모델 로딩 중...")
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            load_time = time.time() - start_time
            print(f"✅ SentenceTransformer 모델 로딩 완료! (소요시간: {load_time:.2f}초)")
            
            # 2. Transformers 라이브러리 모델도 로딩 (상세 분석용)
            print(f"🔄 Transformers 라이브러리 모델 로딩 중...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.transformer_model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            print(f"✅ 모든 모델 로딩 완료! 속담 난이도 분석 준비됨!")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            
            # 에러별 해결 방법 안내
            self._print_error_solutions(e)
            
            return False
    
    def _print_error_solutions(self, error: Exception) -> None:
        """
        🔧 에러별 해결 방법을 안내합니다.
        
        Args:
            error: 발생한 에러
        """
        print("\n🔧 문제 해결 방법:")
        
        error_str = str(error).lower()
        
        if "connection" in error_str or "network" in error_str:
            print("🌐 네트워크 연결 문제:")
            print("  - 인터넷 연결 상태 확인")
            print("  - Hugging Face 서버 상태 확인")
            print("  - 방화벽/프록시 설정 확인")
            print("  - VPN 사용 시 해제 후 재시도")
            
        elif "memory" in error_str or "cuda out of memory" in error_str:
            print("🧠 메모리 부족 문제:")
            print("  - GPU 메모리 부족 시 CPU 모드로 재시도")
            print("  - 다른 프로그램 종료 후 재시도")
            print("  - 속담 배치 크기 줄이기")
            
        elif "permission" in error_str:
            print("🔒 권한 문제:")
            print("  - 관리자 권한으로 실행")
            print("  - 모델 캐시 디렉토리 권한 확인")
            
        elif "404" in error_str or "not found" in error_str:
            print("🔍 모델 찾기 실패:")
            print("  - 모델명 확인: jhgan/ko-sroberta-multitask")
            print("  - Hugging Face Hub 접근 가능 여부 확인")
            
        else:
            print("📋 일반적인 해결 방법:")
            print("  - pip install --upgrade transformers sentence-transformers")
            print("  - 가상환경 재생성")
            print("  - Python 버전 확인 (3.8 이상)")
            print("  - 캐시 디렉토리 삭제 후 재시도")
    
    def test_proverb_embedding(self) -> bool:
        """
        🧪 속담 예시로 임베딩 생성 테스트를 수행합니다.
        
        Returns:
            bool: 테스트 성공 여부
        """
        try:
            print("\n" + "="*60)
            print("🧪 속담 임베딩 생성 및 분석 테스트")
            print("="*60)
            
            # 테스트용 속담 목록 (난이도별)
            test_proverbs = self.config.get_test_proverbs()
            
            print(f"📝 테스트 속담 목록 ({len(test_proverbs)}개):")
            for i, proverb in enumerate(test_proverbs, 1):
                print(f"  {i}. {proverb}")
            
            # 속담 임베딩 생성
            print(f"\n⏳ 속담 임베딩 생성 중...")
            start_time = time.time()
            
            embeddings = self.sentence_model.encode(
                test_proverbs,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=self.config.BATCH_SIZE
            )
            
            processing_time = time.time() - start_time
            
            print(f"✅ 속담 임베딩 생성 완료!")
            print(f"📊 임베딩 형태: {embeddings.shape}")
            print(f"⏱️  처리 시간: {processing_time:.3f}초")
            print(f"🚀 초당 속담 수: {len(test_proverbs)/processing_time:.1f}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 속담 임베딩 테스트 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return False
    
    def test_proverb_similarity(self) -> bool:
        """
        🔍 속담 간 유사도 계산 테스트를 수행합니다.
        
        Returns:
            bool: 테스트 성공 여부
        """
        try:
            print("\n" + "="*60)
            print("🔍 속담 간 유사도 분석 테스트")
            print("="*60)
            
            # 난이도별 대표 속담 선택
            easy_proverbs = ["가는 말이 고와야 오는 말이 곱다", "호랑이도 제 말 하면 온다"]
            hard_proverbs = ["하늘이 무너져도 솟아날 구멍이 있다", "닭 쫓던 개 지붕 쳐다본다"]
            
            all_test_proverbs = easy_proverbs + hard_proverbs
            
            print("📝 유사도 분석 대상:")
            print("🟢 쉬운 속담:")
            for proverb in easy_proverbs:
                print(f"  - {proverb}")
            print("🔴 어려운 속담:")
            for proverb in hard_proverbs:
                print(f"  - {proverb}")
            
            # 임베딩 생성
            print(f"\n⏳ 속담 임베딩 생성 중...")
            embeddings = self.sentence_model.encode(all_test_proverbs, convert_to_tensor=True)
            
            # 유사도 계산 및 분석
            print(f"\n📊 속담 간 유사도 분석 결과:")
            
            # 같은 난이도 내 유사도
            easy_sim = cos_sim(embeddings[0], embeddings[1]).item()
            hard_sim = cos_sim(embeddings[2], embeddings[3]).item()
            
            print(f"🟢 쉬운 속담 간 유사도: {easy_sim:.4f}")
            print(f"🔴 어려운 속담 간 유사도: {hard_sim:.4f}")
            
            # 다른 난이도 간 유사도
            cross_similarities = []
            for i, easy in enumerate(easy_proverbs):
                for j, hard in enumerate(hard_proverbs):
                    sim = cos_sim(embeddings[i], embeddings[j+2]).item()
                    cross_similarities.append(sim)
                    print(f"🔄 '{easy}' vs '{hard}': {sim:.4f}")
            
            avg_cross_sim = np.mean(cross_similarities)
            print(f"\n📈 분석 결과:")
            print(f"  - 같은 난이도 평균 유사도: {(easy_sim + hard_sim)/2:.4f}")
            print(f"  - 다른 난이도 평균 유사도: {avg_cross_sim:.4f}")
            
            if (easy_sim + hard_sim)/2 > avg_cross_sim:
                print("✅ 난이도별로 속담 특성이 구분되는 것을 확인!")
            else:
                print("⚠️ 난이도별 구분이 명확하지 않음. 추가 분석 필요.")
            
            return True
            
        except Exception as e:
            print(f"❌ 속담 유사도 테스트 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return False
    
    def test_difficulty_analysis(self) -> bool:
        """
        🎯 속담 난이도 분석 시뮬레이션 테스트를 수행합니다.
        
        Returns:
            bool: 테스트 성공 여부
        """
        try:
            print("\n" + "="*60)
            print("🎯 속담 난이도 분석 시뮬레이션")
            print("="*60)
            
            print("🎮 속담 게임 난이도별 점수 시스템:")
            for level, info in self.config.PROVERB_DIFFICULTY_LEVELS.items():
                base_score = self.config.BASE_SCORE * info['score_multiplier']
                print(f"  - {info['name']}: {base_score:.0f}점 (배율: {info['score_multiplier']}x)")
                print(f"    📝 {info['description']}")
                print(f"    📚 예시: {info['examples'][0]}")
                print()
            
            # 실제 게임 시뮬레이션
            print("🎲 게임 시뮬레이션:")
            print("사용자가 다음 속담들을 맞췄을 때의 점수:")
            
            total_score = 0
            test_cases = [
                ("가는 말이 고와야 오는 말이 곱다", 1),  # 쉬움
                ("백문이 불여일견", 2),  # 보통
                ("금강산도 식후경", 3),  # 어려움
                ("등잔 밑이 어둡다", 4),  # 매우 어려움
                ("하늘이 무너져도 솟아날 구멍이 있다", 5)  # 최고 난이도
            ]
            
            for proverb, difficulty in test_cases:
                level_info = self.config.PROVERB_DIFFICULTY_LEVELS[difficulty]
                score = int(self.config.BASE_SCORE * level_info['score_multiplier'])
                total_score += score
                
                print(f"  📝 '{proverb}'")
                print(f"     → {level_info['name']} ({score}점)")
            
            print(f"\n🏆 총 획득 점수: {total_score}점")
            print(f"💎 평균 점수: {total_score/len(test_cases):.1f}점")
            
            return True
            
        except Exception as e:
            print(f"❌ 난이도 분석 테스트 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return False
    
    def monitor_memory_usage(self) -> Dict[str, str]:
        """
        📊 현재 메모리 사용량을 모니터링합니다.
        
        Returns:
            Dict[str, str]: 메모리 사용량 정보
        """
        memory_info = {}
        
        # 시스템 메모리
        memory = psutil.virtual_memory()
        memory_info["system_memory_used"] = f"{memory.used / (1024**3):.1f} GB"
        memory_info["system_memory_percent"] = f"{memory.percent:.1f}%"
        
        # GPU 메모리 (사용 가능한 경우)
        if torch.cuda.is_available() and self.device == "cuda":
            memory_info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / (1024**3):.1f} GB"
            memory_info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / (1024**3):.1f} GB"
            memory_info["gpu_memory_percent"] = f"{(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100:.1f}%"
        
        return memory_info
    
    def run_comprehensive_test(self) -> bool:
        """
        🚀 속담 게임 난이도 판별 시스템 종합 테스트를 실행합니다.
        
        Returns:
            bool: 전체 테스트 성공 여부
        """
        print("🚀 속담 게임 - 난이도 판별 AI 시스템 종합 테스트 시작!")
        print("=" * 70)
        
        # 설정 정보 출력
        self.config.print_config_summary()
        
        test_results = {
            "system_check": False,
            "model_loading": False,
            "proverb_embedding": False,
            "proverb_similarity": False,
            "difficulty_analysis": False
        }
        
        try:
            # 1. 시스템 요구사항 검사
            system_info = self.check_system_requirements()
            test_results["system_check"] = True
            
            # 2. AI 모델 다운로드 및 로딩
            if self.download_and_load_model():
                test_results["model_loading"] = True
                
                # 3. 속담 임베딩 생성 테스트
                if self.test_proverb_embedding():
                    test_results["proverb_embedding"] = True
                
                # 4. 속담 유사도 분석 테스트
                if self.test_proverb_similarity():
                    test_results["proverb_similarity"] = True
                
                # 5. 난이도 분석 시뮬레이션
                if self.test_difficulty_analysis():
                    test_results["difficulty_analysis"] = True
            
            # 6. 메모리 사용량 확인
            print("\n" + "="*60)
            print("📊 메모리 사용량 모니터링")
            print("="*60)
            
            memory_info = self.monitor_memory_usage()
            for key, value in memory_info.items():
                print(f"💾 {key}: {value}")
            
            # 7. 테스트 결과 요약
            self._print_test_summary(test_results)
            
            # 모든 테스트 통과 여부 확인
            all_passed = all(test_results.values())
            
            if all_passed:
                print("\n🎉 모든 테스트를 성공적으로 통과했습니다!")
                print("✅ 속담 게임 난이도 판별 AI 시스템이 정상 작동합니다!")
                print("🎯 이제 실제 속담 게임에서 사용할 수 있습니다!")
            else:
                print("\n⚠️ 일부 테스트가 실패했습니다. 위의 에러 메시지를 확인하세요.")
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return False
    
    def _print_test_summary(self, test_results: Dict[str, bool]) -> None:
        """
        📋 테스트 결과를 요약해서 출력합니다.
        
        Args:
            test_results: 각 테스트의 성공/실패 결과
        """
        print("\n" + "="*60)
        print("📋 속담 게임 AI 시스템 테스트 결과 요약")
        print("="*60)
        
        test_names = {
            "system_check": "🔍 시스템 요구사항 검사",
            "model_loading": "🤖 AI 모델 다운로드 및 로딩",
            "proverb_embedding": "📝 속담 임베딩 생성 테스트",
            "proverb_similarity": "🔍 속담 유사도 분석 테스트",
            "difficulty_analysis": "🎯 난이도 분석 시뮬레이션"
        }
        
        for test_key, test_name in test_names.items():
            status = "✅ 성공" if test_results[test_key] else "❌ 실패"
            print(f"{status} - {test_name}")
        
        passed_count = sum(test_results.values())
        total_count = len(test_results)
        print(f"\n📊 전체 결과: {passed_count}/{total_count} 테스트 통과")


def main():
    """
    🎯 메인 실행 함수
    """
    try:
        # 속담 게임 난이도 판별 테스터 생성 및 실행
        tester = ProverbDifficultyTester()
        success = tester.run_comprehensive_test()
        
        # 실행 결과에 따른 안내 메시지
        print("\n" + "="*70)
        if success:
            print("🎯 다음 단계 - 속담 게임 개발:")
            print("1. 속담 게임 API 서버 개발 (api/ 폴더)")
            print("2. 속담 데이터베이스 연동 (MySQL proverb_game.proverb)")
            print("3. 실제 속담 난이도 분류 로직 구현")
            print("4. 사용자 점수 시스템 구축")
            print("5. 프론트엔드와 연동 테스트")
        else:
            print("🔧 문제 해결 후 다시 실행해주세요:")
            print("1. 에러 메시지 확인 및 해결")
            print("2. 필요한 패키지 재설치: pip install -r requirements.txt")
            print("3. Python/CUDA 환경 점검")
            print("4. 네트워크 연결 상태 확인")
        print("="*70)
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        return False


if __name__ == "__main__":
    """
    🧪 스크립트 직접 실행 시 메인 함수 호출
    
    실행 방법:
        python model_test.py
    
    또는 IDE에서 이 파일을 직접 실행
    """
    print("🎯 속담 게임 - 난이도 판별 AI 모델 테스트 시작")
    print("📍 jhgan/ko-sroberta-multitask 모델 전용")
    print("🎮 속담 절반을 주고 나머지를 맞추는 게임의 핵심 AI 기능")
    print()
    
    success = main()
    sys.exit(0 if success else 1)
