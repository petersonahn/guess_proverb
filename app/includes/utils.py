"""
🎯 속담 게임 - 유틸리티 함수 모듈

이 모듈은 속담 난이도 분석 시스템에서 사용되는 
유용한 유틸리티 함수들을 제공합니다.

주요 기능:
1. 속담 텍스트 처리 및 결합
2. 데이터베이스 연결 검증
3. 난이도 분포 계산 및 통계
4. 분석 결과 내보내기
5. 시스템 상태 모니터링

사용 예시:
    from utils import combine_proverb_parts, validate_database_connection
    
    full_proverb = combine_proverb_parts("가는 말이 고와야", "오는 말이 곱다")
    is_connected = validate_database_connection()
"""

import os
import sys
import json
import csv
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging

# config 및 database 모듈 import
try:
    import sys
    import os
    # 현재 파일의 부모 디렉토리들을 sys.path에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # app 디렉토리
    root_dir = os.path.dirname(parent_dir)     # 프로젝트 루트
    sys.path.insert(0, root_dir)
    
    from app.core.config import proverb_config
    from app.includes.dbconn import ProverbDatabase
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)


def combine_proverb_parts(question: str, answer: str, separator: str = " ") -> str:
    """
    🔗 속담의 앞부분과 뒷부분을 합쳐서 완전한 속담을 만듭니다.
    
    Args:
        question: 속담 앞부분 (문제)
        answer: 속담 뒷부분 (정답)
        separator: 연결 구분자 (기본: 공백)
        
    Returns:
        str: 완성된 속담
        
    Example:
        >>> combine_proverb_parts("가는 말이 고와야", "오는 말이 곱다")
        "가는 말이 고와야 오는 말이 곱다"
    """
    if not question or not answer:
        return ""
    
    # 공백 정리
    question = question.strip()
    answer = answer.strip()
    
    if not question or not answer:
        return ""
    
    # 자연스러운 연결을 위한 처리
    if question.endswith(('은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과')):
        return f"{question} {answer}"
    elif question.endswith(('하면', '면', '니', '야')):
        return f"{question} {answer}"
    else:
        return f"{question}{separator}{answer}"


def validate_database_connection() -> Dict[str, Union[bool, str, int]]:
    """
    🗄️ 데이터베이스 연결 유효성을 검사합니다.
    
    Returns:
        Dict: 연결 상태 정보
        {
            "success": True/False,
            "connected": True/False,
            "message": "연결 상태 메시지",
            "host": "localhost",
            "database": "proverb_game",
            "table_exists": True/False,
            "proverb_count": 90
        }
    """
    result = {
        "success": False,
        "connected": False,
        "message": "",
        "host": proverb_config.DB_HOST,
        "database": proverb_config.DB_NAME,
        "table_exists": False,
        "proverb_count": 0
    }
    
    try:
        print("🔍 데이터베이스 연결 검증 중...")
        
        # 데이터베이스 연결 시도
        db = ProverbDatabase()
        
        # 연결 테스트
        if not db.test_connection():
            result["message"] = "데이터베이스 연결 실패"
            return result
        
        result["success"] = True
        result["connected"] = True
        
        # 테이블 존재 확인
        table_exists = db.check_table_exists()
        result["table_exists"] = table_exists
        
        if table_exists:
            # 속담 개수 확인
            proverb_count = db.get_proverb_count()
            result["proverb_count"] = proverb_count
            result["message"] = f"연결 성공 - {proverb_count}개 속담 확인"
        else:
            result["message"] = "연결 성공 - proverb 테이블 없음"
        
        db.close()
        
        print(f"✅ {result['message']}")
        return result
        
    except Exception as e:
        result["message"] = f"검증 실패: {str(e)}"
        print(f"❌ {result['message']}")
        return result


def calculate_difficulty_distribution(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    📊 난이도 분석 결과의 분포를 계산합니다.
    
    Args:
        analysis_results: 난이도 분석 결과 목록
        
    Returns:
        Dict: 난이도 분포 통계
        {
            "total_count": 90,
            "success_count": 87,
            "difficulty_distribution": {
                1: {"count": 30, "percentage": 34.5, "total_score": 30},
                2: {"count": 35, "percentage": 40.2, "total_score": 70}, 
                3: {"count": 22, "percentage": 25.3, "total_score": 66}
            },
            "average_difficulty": 1.9,
            "total_possible_score": 166,
            "confidence_stats": {
                "average": 0.72,
                "min": 0.45,
                "max": 0.95
            }
        }
    """
    if not analysis_results:
        return {"error": "분석 결과가 없습니다"}
    
    print("📊 난이도 분포 계산 중...")
    
    # 성공적인 분석 결과만 필터링
    success_results = [r for r in analysis_results if r.get('difficulty_level', 0) > 0]
    
    # 기본 통계
    total_count = len(analysis_results)
    success_count = len(success_results)
    
    if success_count == 0:
        return {
            "total_count": total_count,
            "success_count": 0,
            "error": "성공적으로 분석된 결과가 없습니다"
        }
    
    # 난이도별 분포 계산
    difficulty_distribution = {}
    difficulty_levels = [1, 2, 3]
    confidence_scores = []
    total_score = 0
    difficulty_sum = 0
    
    for level in difficulty_levels:
        level_results = [r for r in success_results if r.get('difficulty_level') == level]
        count = len(level_results)
        percentage = (count / success_count) * 100 if success_count > 0 else 0
        level_total_score = count * level  # 각 레벨의 점수는 레벨 번호와 같음
        
        difficulty_distribution[level] = {
            "count": count,
            "percentage": round(percentage, 1),
            "total_score": level_total_score
        }
        
        total_score += level_total_score
        difficulty_sum += level * count
        
        # 해당 레벨의 신뢰도 점수 수집
        level_confidences = [r.get('confidence', 0) for r in level_results]
        confidence_scores.extend(level_confidences)
    
    # 평균 난이도 계산
    average_difficulty = difficulty_sum / success_count if success_count > 0 else 0
    
    # 신뢰도 통계
    confidence_stats = {}
    if confidence_scores:
        confidence_stats = {
            "average": round(sum(confidence_scores) / len(confidence_scores), 3),
            "min": round(min(confidence_scores), 3),
            "max": round(max(confidence_scores), 3)
        }
    
    distribution = {
        "total_count": total_count,
        "success_count": success_count,
        "success_rate": round((success_count / total_count) * 100, 1) if total_count > 0 else 0,
        "difficulty_distribution": difficulty_distribution,
        "average_difficulty": round(average_difficulty, 2),
        "total_possible_score": total_score,
        "confidence_stats": confidence_stats
    }
    
    print(f"✅ 난이도 분포 계산 완료:")
    print(f"  - 전체: {total_count}개, 성공: {success_count}개 ({distribution['success_rate']}%)")
    print(f"  - 평균 난이도: {average_difficulty:.2f}")
    print(f"  - 총 점수: {total_score}점")
    
    return distribution


def export_analysis_results(results: List[Dict[str, Any]], 
                          output_format: str = "json",
                          output_path: Optional[str] = None) -> str:
    """
    📤 분석 결과를 파일로 내보냅니다.
    
    Args:
        results: 분석 결과 목록
        output_format: 출력 형식 ("json", "csv")
        output_path: 출력 파일 경로 (None이면 자동 생성)
        
    Returns:
        str: 생성된 파일 경로
        
    Example:
        >>> export_path = export_analysis_results(results, "json")
        >>> print(f"결과를 {export_path}에 저장했습니다")
    """
    if not results:
        raise ValueError("내보낼 결과가 없습니다")
    
    # 출력 디렉토리 설정
    output_dir = os.path.join(proverb_config.BASE_DIR, "exports")
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 자동 생성
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"proverb_analysis_{timestamp}.{output_format.lower()}"
        output_path = os.path.join(output_dir, filename)
    
    print(f"📤 분석 결과 내보내기: {output_format.upper()} 형식")
    print(f"📁 출력 경로: {output_path}")
    
    try:
        if output_format.lower() == "json":
            # JSON 형식으로 내보내기
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_count": len(results),
                    "success_count": len([r for r in results if r.get('difficulty_level', 0) > 0]),
                    "format": "json"
                },
                "difficulty_levels": proverb_config.PROVERB_DIFFICULTY_LEVELS,
                "analysis_results": results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        elif output_format.lower() == "csv":
            # CSV 형식으로 내보내기
            fieldnames = [
                'proverb_id', 'full_proverb', 'difficulty_level', 
                'confidence', 'score', 'processing_time', 'message'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(row)
        
        else:
            raise ValueError(f"지원하지 않는 형식: {output_format}")
        
        file_size = os.path.getsize(output_path)
        print(f"✅ 내보내기 완료: {output_path} ({file_size:,} bytes)")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 내보내기 실패: {str(e)}")
        raise


def print_analysis_summary_table(results: List[Dict[str, Any]], 
                                max_rows: int = 20) -> None:
    """
    📋 분석 결과를 테이블 형태로 출력합니다.
    
    Args:
        results: 분석 결과 목록
        max_rows: 최대 출력 행 수
    """
    if not results:
        print("❌ 출력할 결과가 없습니다")
        return
    
    # 성공적인 결과만 필터링
    success_results = [r for r in results if r.get('difficulty_level', 0) > 0]
    
    if not success_results:
        print("❌ 성공적으로 분석된 결과가 없습니다")
        return
    
    print(f"\n📋 분석 결과 요약 테이블 (상위 {min(max_rows, len(success_results))}개)")
    print("=" * 100)
    
    # 헤더 출력
    header = f"{'ID':<4} {'속담':<40} {'난이도':<10} {'점수':<4} {'신뢰도':<8} {'시간':<8}"
    print(header)
    print("-" * 100)
    
    # 데이터 출력
    for i, result in enumerate(success_results[:max_rows]):
        proverb_id = result.get('proverb_id', 'N/A')
        full_proverb = result.get('full_proverb', '')
        
        # 속담 텍스트 길이 제한
        if len(full_proverb) > 35:
            proverb_display = full_proverb[:32] + "..."
        else:
            proverb_display = full_proverb
        
        difficulty_level = result.get('difficulty_level', 0)
        level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(difficulty_level, {})
        level_name = level_info.get('name', '알 수 없음')
        
        score = result.get('score', 0)
        confidence = result.get('confidence', 0.0)
        processing_time = result.get('processing_time', 0.0)
        
        row = f"{proverb_id:<4} {proverb_display:<40} {level_name:<10} {score:<4} {confidence:<8.1%} {processing_time:<8.3f}"
        print(row)
    
    # 요약 통계
    if len(success_results) > max_rows:
        print(f"\n... 및 {len(success_results) - max_rows}개 더")
    
    # 분포 요약
    distribution = calculate_difficulty_distribution(results)
    print(f"\n📊 요약:")
    print(f"  - 전체: {distribution['total_count']}개")
    print(f"  - 성공: {distribution['success_count']}개 ({distribution['success_rate']}%)")
    print(f"  - 평균 난이도: {distribution['average_difficulty']}")
    print(f"  - 총 점수: {distribution['total_possible_score']}점")
    print("=" * 100)


def get_system_status() -> Dict[str, Any]:
    """
    🖥️ 시스템 상태 정보를 조회합니다.
    
    Returns:
        Dict: 시스템 상태 정보
    """
    try:
        import psutil
        import torch
        
        # 시스템 정보
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total": f"{memory.total / (1024**3):.1f} GB",
                "memory_used": f"{memory.used / (1024**3):.1f} GB",
                "memory_percent": memory.percent
            },
            "ai_model": {
                "name": proverb_config.MODEL_NAME,
                "device": proverb_config.DEVICE,
                "cache_dir": proverb_config.MODEL_CACHE_DIR
            },
            "database": validate_database_connection(),
            "config": {
                "batch_size": proverb_config.BATCH_SIZE_ANALYSIS,
                "caching_enabled": proverb_config.ENABLE_CACHING,
                "difficulty_levels": len(proverb_config.PROVERB_DIFFICULTY_LEVELS)
            }
        }
        
        # GPU 정보 (사용 가능한 경우)
        if torch.cuda.is_available():
            status["system"]["gpu_available"] = True
            status["system"]["gpu_name"] = torch.cuda.get_device_name(0)
            status["system"]["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB"
        else:
            status["system"]["gpu_available"] = False
        
        return status
        
    except Exception as e:
        return {
            "error": f"시스템 상태 조회 실패: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


def cleanup_cache_files(cache_dir: Optional[str] = None, 
                       max_age_days: int = 30) -> Dict[str, Any]:
    """
    🗑️ 오래된 캐시 파일들을 정리합니다.
    
    Args:
        cache_dir: 캐시 디렉토리 (None이면 기본 캐시 디렉토리 사용)
        max_age_days: 최대 보관 일수
        
    Returns:
        Dict: 정리 결과
    """
    if not cache_dir:
        cache_dir = proverb_config.MODEL_CACHE_DIR
    
    if not os.path.exists(cache_dir):
        return {"message": "캐시 디렉토리가 존재하지 않습니다"}
    
    import time
    from pathlib import Path
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    cleaned_files = []
    total_size_freed = 0
    
    try:
        for file_path in Path(cache_dir).rglob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    
                    cleaned_files.append({
                        "path": str(file_path),
                        "size": file_size,
                        "age_days": file_age / (24 * 60 * 60)
                    })
                    
                    total_size_freed += file_size
        
        result = {
            "cleaned_files_count": len(cleaned_files),
            "total_size_freed": f"{total_size_freed / (1024**2):.1f} MB",
            "max_age_days": max_age_days,
            "cache_dir": cache_dir
        }
        
        if cleaned_files:
            print(f"🗑️ 캐시 정리 완료: {len(cleaned_files)}개 파일, {result['total_size_freed']} 확보")
        else:
            print(f"✅ 정리할 캐시 파일이 없습니다")
        
        return result
        
    except Exception as e:
        return {"error": f"캐시 정리 실패: {str(e)}"}


def test_all_utilities():
    """
    🧪 모든 유틸리티 함수들을 테스트합니다.
    """
    print("🧪 유틸리티 함수 종합 테스트 시작")
    print("=" * 60)
    
    test_results = []
    
    # 1. 속담 결합 테스트
    try:
        print("\n1. 속담 결합 테스트:")
        combined = combine_proverb_parts("가는 말이 고와야", "오는 말이 곱다")
        print(f"   결과: '{combined}'")
        test_results.append(("속담 결합", True, ""))
    except Exception as e:
        print(f"   ❌ 실패: {str(e)}")
        test_results.append(("속담 결합", False, str(e)))
    
    # 2. 데이터베이스 연결 검증 테스트
    try:
        print("\n2. 데이터베이스 연결 검증:")
        db_status = validate_database_connection()
        print(f"   연결 상태: {db_status['connected']}")
        print(f"   메시지: {db_status['message']}")
        test_results.append(("DB 연결 검증", db_status['connected'], db_status['message']))
    except Exception as e:
        print(f"   ❌ 실패: {str(e)}")
        test_results.append(("DB 연결 검증", False, str(e)))
    
    # 3. 시스템 상태 조회 테스트
    try:
        print("\n3. 시스템 상태 조회:")
        system_status = get_system_status()
        if 'error' not in system_status:
            print(f"   CPU 사용률: {system_status['system']['cpu_percent']}%")
            print(f"   메모리 사용률: {system_status['system']['memory_percent']}%")
            print(f"   AI 모델: {system_status['ai_model']['name']}")
            test_results.append(("시스템 상태", True, ""))
        else:
            print(f"   ❌ {system_status['error']}")
            test_results.append(("시스템 상태", False, system_status['error']))
    except Exception as e:
        print(f"   ❌ 실패: {str(e)}")
        test_results.append(("시스템 상태", False, str(e)))
    
    # 4. 테스트 결과 요약
    print(f"\n📊 테스트 결과 요약:")
    print("-" * 40)
    
    success_count = 0
    for test_name, success, message in test_results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{status} - {test_name}")
        if not success and message:
            print(f"        {message}")
        if success:
            success_count += 1
    
    print(f"\n전체 결과: {success_count}/{len(test_results)} 테스트 통과")
    
    return success_count == len(test_results)


if __name__ == "__main__":
    """
    🚀 스크립트 직접 실행 시 테스트 함수 호출
    
    실행 방법:
        python utils.py
    """
    print("🛠️ 속담 게임 - 유틸리티 함수 테스트")
    print("데이터베이스 연결, 텍스트 처리, 시스템 상태 등")
    print()
    
    success = test_all_utilities()
    sys.exit(0 if success else 1)
