"""
🎯 속담 게임 - 데이터베이스 연동 속담 난이도 분석 클래스

이 모듈은 데이터베이스에 저장된 속담의 난이도를 자동으로 분석하는 핵심 클래스를 제공합니다.
- 데이터베이스의 속담을 3단계 난이도로 분류 (1점, 2점, 3점)
- jhgan/ko-sroberta-multitask 모델 사용
- 배치 처리 및 캐싱으로 성능 최적화

주요 기능:
1. 데이터베이스 연동 (proverb_game.proverb 테이블)
2. AI 모델을 통한 난이도 분석
3. 배치 처리 (16개씩 묶어서 처리)
4. 분석 결과 캐싱 및 성능 모니터링
5. 진행률 표시 (tqdm)

사용 예시:
    analyzer = ProverbDifficultyAnalyzer()
    result = analyzer.analyze_proverb_difficulty(proverb_id=1)
    all_results = analyzer.batch_analyze_all_proverbs()
"""

import os
import sys
import time
import traceback
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import numpy as np

# config 및 database 모듈 import
try:
    from config import proverb_config
    from database import ProverbDatabase
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("해결 방법: config.py, database.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

# 필수 라이브러리 import
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ 필수 라이브러리 import 실패: {e}")
    print("해결 방법: pip install -r requirements.txt")
    sys.exit(1)


class ProverbDifficultyAnalyzer:
    """
    🎯 속담 게임 - 데이터베이스 연동 난이도 분석 클래스
    
    데이터베이스에 저장된 속담의 난이도를 AI 모델로 자동 분석합니다.
    
    특징:
    - 데이터베이스 연동 (proverb_game.proverb 테이블)
    - jhgan/ko-sroberta-multitask 모델 전용
    - 3단계 난이도 분류 (1점, 2점, 3점)
    - 배치 처리로 성능 최적화 (16개씩 처리)
    - 분석 결과 캐싱으로 중복 분석 방지
    """
    
    def __init__(self):
        """
        🤖 속담 난이도 분석기 초기화
        
        1-2단계에서 검증된 모델 로딩 코드를 재사용하고
        데이터베이스 연결을 초기화합니다.
        """
        self.config = proverb_config
        self.model_name = self.config.MODEL_NAME
        self.device = self.config.DEVICE
        self.cache_dir = self.config.MODEL_CACHE_DIR
        self.batch_size = self.config.BATCH_SIZE_ANALYSIS
        
        # 모델 관련 변수
        self.sentence_model = None
        self.tokenizer = None
        self.transformer_model = None
        
        # 데이터베이스 연결
        self.db = None
        
        # 분석 결과 캐시
        self.analysis_cache = {} if self.config.ENABLE_CACHING else None
        
        # 난이도 분석을 위한 참조 속담들
        self.reference_proverbs = self._get_reference_proverbs()
        
        # 성능 통계
        self.stats = {
            "total_analyzed": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
            "batch_count": 0
        }
        
        print(f"🎯 속담 게임 난이도 분석기 초기화 중...")
        print(f"📍 모델: {self.model_name}")
        print(f"💻 디바이스: {self.device}")
        print(f"📦 배치 크기: {self.batch_size}")
        print(f"💾 캐싱: {'활성화' if self.config.ENABLE_CACHING else '비활성화'}")
        
        # 데이터베이스 연결
        if not self._connect_database():
            raise Exception("데이터베이스 연결 실패")
        
        # 모델 로딩
        if not self._load_models():
            raise Exception("AI 모델 로딩 실패")
        
        print(f"✅ 속담 난이도 분석기 준비 완료!")
    
    def _get_reference_proverbs(self) -> Dict[int, List[str]]:
        """
        🧪 각 난이도별 참조 속담들을 가져옵니다.
        
        Returns:
            Dict[int, List[str]]: 난이도별 참조 속담 목록
        """
        reference_proverbs = {}
        for level, info in self.config.PROVERB_DIFFICULTY_LEVELS.items():
            reference_proverbs[level] = info['examples']
        return reference_proverbs
    
    def _connect_database(self) -> bool:
        """
        🗄️ 데이터베이스 연결을 설정합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            print("⏳ 데이터베이스 연결 중...")
            self.db = ProverbDatabase()
            
            # 연결 테스트
            if not self.db.test_connection():
                print("❌ 데이터베이스 연결 테스트 실패")
                return False
            
            # 테이블 존재 확인
            if not self.db.check_table_exists():
                print("❌ proverb 테이블이 존재하지 않습니다")
                return False
            
            proverb_count = self.db.get_proverb_count()
            print(f"✅ 데이터베이스 연결 성공! (속담 {proverb_count}개)")
            return True
            
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {str(e)}")
            return False
    
    def _load_models(self) -> bool:
        """
        🤖 AI 모델들을 로딩합니다.
        
        Returns:
            bool: 모델 로딩 성공 여부
        """
        try:
            print("⏳ AI 모델 로딩 중...")
            
            # 캐시 디렉토리 생성
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # SentenceTransformer 모델 로딩
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Transformers 모델 로딩 (호환성용)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.transformer_model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            print("✅ 모든 AI 모델 로딩 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return False
    
    def _calculate_difficulty_from_embeddings(self, proverb_embedding: np.ndarray) -> Dict[str, float]:
        """
        🧮 임베딩 벡터를 바탕으로 난이도를 계산합니다.
        
        Args:
            proverb_embedding: 속담의 임베딩 벡터
            
        Returns:
            Dict[str, float]: 각 난이도별 확률 점수
        """
        try:
            difficulty_scores = {}
            
            for level, reference_list in self.reference_proverbs.items():
                # 참조 속담들의 임베딩 생성
                reference_embeddings = self.sentence_model.encode(
                    reference_list, 
                    convert_to_tensor=True
                )
                
                # 코사인 유사도 계산
                from sentence_transformers.util import cos_sim
                similarities = cos_sim(proverb_embedding, reference_embeddings)
                
                # 평균 유사도를 해당 난이도 점수로 사용
                avg_similarity = similarities.mean().item()
                difficulty_scores[level] = max(0.0, avg_similarity)
            
            # 점수 정규화
            total_score = sum(difficulty_scores.values())
            if total_score > 0:
                for level in difficulty_scores:
                    difficulty_scores[level] /= total_score
            else:
                # 모든 점수가 0인 경우 균등 분배
                for level in difficulty_scores:
                    difficulty_scores[level] = 1.0 / len(difficulty_scores)
            
            return difficulty_scores
            
        except Exception as e:
            print(f"⚠️ 난이도 계산 중 오류: {str(e)}")
            # 에러 시 균등 분배
            num_levels = len(self.reference_proverbs)
            return {level: 1.0/num_levels for level in self.reference_proverbs.keys()}
    
    def analyze_proverb_difficulty(self, proverb_id: Optional[int] = None, 
                                 proverb_text: Optional[str] = None) -> Dict[str, Any]:
        """
        🎯 속담의 난이도를 분석합니다.
        
        Args:
            proverb_id: 데이터베이스 속담 ID (우선순위)
            proverb_text: 직접 입력한 속담 텍스트
            
        Returns:
            Dict: 난이도 분석 결과
            {
                "proverb_id": 1,
                "full_proverb": "가는 말이 고와야 오는 말이 곱다", 
                "difficulty_level": 2,
                "confidence": 0.85,
                "score": 2,
                "processing_time": 0.15,
                "message": "분석 완료"
            }
        """
        start_time = time.time()
        
        try:
            # 1. 속담 데이터 준비
            if proverb_id is not None:
                # 데이터베이스에서 조회
                proverb_data = self.db.get_proverb_by_id(proverb_id)
                if not proverb_data:
                    return {
                        "proverb_id": proverb_id,
                        "full_proverb": "",
                        "difficulty_level": 0,
                        "confidence": 0.0,
                        "score": 0,
                        "processing_time": time.time() - start_time,
                        "message": f"속담 ID {proverb_id}를 찾을 수 없습니다"
                    }
                
                full_proverb = proverb_data['full_proverb']
                actual_proverb_id = proverb_data['id']
                
            elif proverb_text:
                # 직접 입력된 텍스트 사용
                full_proverb = proverb_text.strip()
                actual_proverb_id = None
                
            else:
                return {
                    "proverb_id": None,
                    "full_proverb": "",
                    "difficulty_level": 0,
                    "confidence": 0.0,
                    "score": 0,
                    "processing_time": time.time() - start_time,
                    "message": "속담 ID 또는 텍스트를 입력해주세요"
                }
            
            # 2. 캐시 확인
            cache_key = f"id_{actual_proverb_id}" if actual_proverb_id else f"text_{hash(full_proverb)}"
            
            if self.analysis_cache and cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key].copy()
                cached_result["processing_time"] = time.time() - start_time
                cached_result["message"] += " (캐시됨)"
                self.stats["cache_hits"] += 1
                return cached_result
            
            print(f"🔍 속담 난이도 분석: '{full_proverb}'")
            
            # 3. 임베딩 생성
            proverb_embedding = self.sentence_model.encode([full_proverb], convert_to_tensor=True)[0]
            
            # 4. 난이도 계산
            difficulty_scores = self._calculate_difficulty_from_embeddings(proverb_embedding)
            
            # 5. 최종 난이도 결정
            predicted_level = max(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])
            confidence = difficulty_scores[predicted_level]
            
            # 6. 점수 계산
            level_info = self.config.PROVERB_DIFFICULTY_LEVELS[predicted_level]
            score = level_info["score"]
            
            # 7. 결과 정리
            processing_time = time.time() - start_time
            
            result = {
                "proverb_id": actual_proverb_id,
                "full_proverb": full_proverb,
                "difficulty_level": predicted_level,
                "confidence": round(confidence, 3),
                "score": score,
                "processing_time": round(processing_time, 3),
                "message": f"분석 완료 - {level_info['name']} ({score}점)"
            }
            
            # 8. 캐시 저장
            if self.analysis_cache:
                self.analysis_cache[cache_key] = result.copy()
            
            # 9. 통계 업데이트
            self.stats["total_analyzed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            print(f"✅ 분석 완료: {level_info['name']} (신뢰도: {confidence:.1%}, {processing_time:.3f}초)")
            
            return result
            
        except Exception as e:
            error_message = f"분석 중 오류 발생: {str(e)}"
            print(f"❌ {error_message}")
            
            return {
                "proverb_id": proverb_id,
                "full_proverb": proverb_text or "",
                "difficulty_level": 0,
                "confidence": 0.0,
                "score": 0,
                "processing_time": time.time() - start_time,
                "message": error_message
            }
    
    def batch_analyze_all_proverbs(self) -> List[Dict[str, Any]]:
        """
        📊 데이터베이스의 모든 속담을 배치 처리로 분석합니다.
        
        16개씩 묶어서 처리하여 메모리 효율성을 확보하고,
        진행률을 표시합니다.
        
        Returns:
            List[Dict]: 모든 속담의 분석 결과
        """
        print("📊 전체 속담 배치 분석 시작")
        print("=" * 60)
        
        start_time = time.time()
        all_results = []
        
        try:
            # 전체 속담 개수 조회
            total_proverbs = self.db.get_proverb_count()
            if total_proverbs == 0:
                print("❌ 분석할 속담이 없습니다")
                return []
            
            print(f"📚 총 {total_proverbs}개 속담 분석 예정")
            print(f"📦 배치 크기: {self.batch_size}개")
            
            # 배치 개수 계산
            total_batches = (total_proverbs + self.batch_size - 1) // self.batch_size
            print(f"🔄 총 {total_batches}개 배치로 처리")
            
            # 진행률 바 설정
            with tqdm(total=total_proverbs, desc="속담 분석 진행", unit="개") as pbar:
                
                for batch_idx in range(total_batches):
                    offset = batch_idx * self.batch_size
                    
                    # 배치 데이터 조회
                    batch_proverbs = self.db.get_proverbs_batch(offset, self.batch_size)
                    
                    if not batch_proverbs:
                        break
                    
                    print(f"\n[배치 {batch_idx + 1}/{total_batches}] {len(batch_proverbs)}개 속담 처리 중...")
                    
                    # 배치 내 각 속담 분석
                    batch_results = []
                    for proverb_data in batch_proverbs:
                        result = self.analyze_proverb_difficulty(proverb_id=proverb_data['id'])
                        batch_results.append(result)
                        pbar.update(1)
                    
                    all_results.extend(batch_results)
                    self.stats["batch_count"] += 1
                    
                    # 배치 완료 정보
                    success_count = sum(1 for r in batch_results if r['difficulty_level'] > 0)
                    print(f"✅ 배치 {batch_idx + 1} 완료: {success_count}/{len(batch_results)}개 성공")
            
            # 전체 결과 통계
            total_time = time.time() - start_time
            success_results = [r for r in all_results if r['difficulty_level'] > 0]
            
            print(f"\n" + "=" * 60)
            print(f"📈 배치 분석 완료 결과:")
            print(f"  - 총 처리 시간: {total_time:.2f}초")
            print(f"  - 성공적으로 분석된 속담: {len(success_results)}/{len(all_results)}개")
            print(f"  - 평균 처리 시간: {total_time/len(all_results):.3f}초/개")
            print(f"  - 캐시 적중률: {self.stats['cache_hits']}/{self.stats['total_analyzed']} ({self.stats['cache_hits']/max(1,self.stats['total_analyzed'])*100:.1f}%)")
            
            # 난이도별 분포
            self._print_difficulty_distribution(success_results)
            
            return all_results
            
        except Exception as e:
            print(f"❌ 배치 분석 실패: {str(e)}")
            print(f"📋 에러 상세: {traceback.format_exc()}")
            return all_results
    
    def _print_difficulty_distribution(self, results: List[Dict[str, Any]]) -> None:
        """
        📊 난이도 분포를 출력합니다.
        
        Args:
            results: 분석 결과 목록
        """
        if not results:
            return
        
        # 난이도별 개수 계산
        difficulty_counts = {1: 0, 2: 0, 3: 0}
        total_score = 0
        
        for result in results:
            level = result.get('difficulty_level', 0)
            if level in difficulty_counts:
                difficulty_counts[level] += 1
                total_score += result.get('score', 0)
        
        total_count = len(results)
        
        print(f"\n📊 난이도 분포:")
        for level, count in difficulty_counts.items():
            level_info = self.config.PROVERB_DIFFICULTY_LEVELS[level]
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            print(f"  - {level_info['name']}: {count}개 ({percentage:.1f}%)")
        
        avg_score = total_score / total_count if total_count > 0 else 0
        print(f"\n🎯 평균 점수: {avg_score:.2f}점")
        print(f"🏆 총 획득 가능 점수: {total_score}점")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        📈 분석 통계 정보를 반환합니다.
        
        Returns:
            Dict: 통계 정보
        """
        return {
            "total_analyzed": self.stats["total_analyzed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_analyzed"]),
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": self.stats["total_processing_time"] / max(1, self.stats["total_analyzed"]),
            "batch_count": self.stats["batch_count"],
            "cache_size": len(self.analysis_cache) if self.analysis_cache else 0,
            "model_info": {
                "name": self.model_name,
                "device": self.device,
                "batch_size": self.batch_size
            }
        }
    
    def clear_cache(self):
        """💾 분석 결과 캐시를 초기화합니다."""
        if self.analysis_cache:
            cache_size = len(self.analysis_cache)
            self.analysis_cache.clear()
            print(f"🗑️ 캐시 초기화 완료 ({cache_size}개 항목 삭제)")
    
    def close(self):
        """🔒 리소스를 정리하고 연결을 종료합니다."""
        if self.db:
            self.db.close()
        
        # 캐시 정리
        if self.analysis_cache:
            self.analysis_cache.clear()
        
        print("✅ 속담 난이도 분석기 종료")


def test_difficulty_analyzer():
    """
    🧪 속담 난이도 분석기 종합 테스트
    """
    print("🧪 속담 난이도 분석기 종합 테스트 시작")
    print("=" * 60)
    
    try:
        # 분석기 초기화
        analyzer = ProverbDifficultyAnalyzer()
        
        # 1. 개별 속담 분석 테스트
        print(f"\n🔍 개별 속담 분석 테스트:")
        print("-" * 40)
        
        # ID로 분석
        result1 = analyzer.analyze_proverb_difficulty(proverb_id=1)
        if result1['difficulty_level'] > 0:
            print(f"✅ ID 분석 성공:")
            print(f"   속담: {result1['full_proverb']}")
            print(f"   난이도: {result1['difficulty_level']}단계 ({result1['score']}점)")
            print(f"   신뢰도: {result1['confidence']:.1%}")
            print(f"   처리시간: {result1['processing_time']:.3f}초")
        
        # 텍스트로 분석
        result2 = analyzer.analyze_proverb_difficulty(proverb_text="호랑이도 제 말 하면 온다")
        if result2['difficulty_level'] > 0:
            print(f"\n✅ 텍스트 분석 성공:")
            print(f"   속담: {result2['full_proverb']}")
            print(f"   난이도: {result2['difficulty_level']}단계 ({result2['score']}점)")
            print(f"   신뢰도: {result2['confidence']:.1%}")
        
        # 2. 배치 분석 테스트 (처음 5개만)
        print(f"\n📦 배치 분석 테스트 (처음 5개 속담):")
        print("-" * 40)
        
        # 임시로 배치 크기를 5로 설정
        original_batch_size = analyzer.batch_size
        analyzer.batch_size = 5
        
        batch_results = []
        batch_proverbs = analyzer.db.get_proverbs_batch(0, 5)
        
        for proverb_data in batch_proverbs:
            result = analyzer.analyze_proverb_difficulty(proverb_id=proverb_data['id'])
            batch_results.append(result)
        
        # 결과 테이블 형태로 출력
        print(f"\n📋 분석 결과 테이블:")
        print(f"{'ID':<4} {'속담':<30} {'난이도':<8} {'점수':<4} {'신뢰도':<8}")
        print("-" * 60)
        
        for result in batch_results:
            if result['difficulty_level'] > 0:
                proverb_short = result['full_proverb'][:25] + "..." if len(result['full_proverb']) > 25 else result['full_proverb']
                level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS[result['difficulty_level']]
                print(f"{result['proverb_id']:<4} {proverb_short:<30} {level_info['name']:<8} {result['score']:<4} {result['confidence']:.1%}")
        
        # 배치 크기 복원
        analyzer.batch_size = original_batch_size
        
        # 3. 통계 정보 출력
        print(f"\n📈 분석 통계:")
        print("-" * 40)
        stats = analyzer.get_analysis_statistics()
        print(f"총 분석 횟수: {stats['total_analyzed']}개")
        print(f"캐시 적중: {stats['cache_hits']}회 ({stats['cache_hit_rate']:.1%})")
        print(f"평균 처리시간: {stats['average_processing_time']:.3f}초")
        print(f"캐시 크기: {stats['cache_size']}개")
        
        # 4. 정리
        analyzer.close()
        
        print(f"\n✅ 속담 난이도 분석기 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        print(f"📋 에러 상세: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    """
    🚀 스크립트 직접 실행 시 테스트 함수 호출
    
    실행 방법:
        python difficulty_analyzer_new.py
    """
    print("🎯 속담 게임 - 데이터베이스 연동 난이도 분석기")
    print("jhgan/ko-sroberta-multitask 모델 사용")
    print("3단계 난이도 분류 + 배치 처리 + 캐싱")
    print("데이터베이스: proverb_game.proverb (root/0000)")
    print()
    
    success = test_difficulty_analyzer()
    sys.exit(0 if success else 1)
