"""
🎯 속말 게임 - 유틸리티 함수들

난이도 분석 결과 표시, 테이블 출력, 통계 계산 등의 유틸리티 함수들을 제공합니다.
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime

# config 모듈 import
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # app 디렉토리
    root_dir = os.path.dirname(parent_dir)     # 프로젝트 루트
    sys.path.insert(0, root_dir)
    
    from app.core.config import proverb_config
except ImportError as e:
    print(f"❌ config 모듈 import 실패: {e}")
    sys.exit(1)


def print_analysis_summary_table(results: List[Dict[str, Any]], max_rows: int = None) -> None:
    """
    📋 난이도 분석 결과를 요약 테이블 형태로 출력합니다.
    
    Args:
        results: 분석 결과 목록
        max_rows: 최대 표시할 행 수
    """
    if not results:
        print("❌ 표시할 분석 결과가 없습니다.")
        return
    
    # 성공적으로 분석된 결과만 필터링
    successful_results = [r for r in results if r.get('difficulty_level', 0) > 0]
    
    if not successful_results:
        print("❌ 성공적으로 분석된 결과가 없습니다.")
        return
    
    # 표시할 결과 제한
    if max_rows:
        display_results = successful_results[:max_rows]
    else:
        display_results = successful_results
    
    # 테이블 헤더
    print(f"{'ID':<4} {'속담':<35} {'난이도':<10} {'점수':<4} {'신뢰도':<8} {'시간':<8}")
    print("-" * 75)
    
    # 테이블 내용
    for result in display_results:
        proverb_id = result.get('proverb_id', 0)
        full_proverb = result.get('full_proverb', '')
        difficulty_level = result.get('difficulty_level', 0)
        score = result.get('score', 0)
        confidence = result.get('confidence', 0.0)
        processing_time = result.get('processing_time', 0.0)
        
        # 속담 텍스트 길이 제한
        if len(full_proverb) > 32:
            proverb_display = full_proverb[:29] + "..."
        else:
            proverb_display = full_proverb
        
        # 난이도 이름
        level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(difficulty_level, {})
        level_name = level_info.get('name', f'{difficulty_level}단계')
        
        print(f"{proverb_id:<4} {proverb_display:<35} {level_name:<10} {score:<4} {confidence:.1%:<8} {processing_time:.3f}s")
    
    # 요약 정보
    if max_rows and len(successful_results) > max_rows:
        print(f"... (총 {len(successful_results)}개 중 {max_rows}개 표시)")


def print_detailed_analysis_table(results: List[Dict[str, Any]], max_rows: int = None) -> None:
    """
    📊 난이도 분석 결과를 상세 테이블 형태로 출력합니다.
    
    Args:
        results: 분석 결과 목록
        max_rows: 최대 표시할 행 수
    """
    if not results:
        print("❌ 표시할 분석 결과가 없습니다.")
        return
    
    # 성공적으로 분석된 결과만 필터링
    successful_results = [r for r in results if r.get('difficulty_level', 0) > 0]
    
    if not successful_results:
        print("❌ 성공적으로 분석된 결과가 없습니다.")
        return
    
    # 표시할 결과 제한
    if max_rows:
        display_results = successful_results[:max_rows]
    else:
        display_results = successful_results
    
    # 상세 테이블 헤더
    print(f"{'ID':<4} {'속담':<30} {'난이도':<8} {'점수':<4} {'신뢰도':<8} {'언어학':<8} {'AI':<8} {'시간':<8}")
    print("-" * 85)
    
    # 테이블 내용
    for result in display_results:
        proverb_id = result.get('proverb_id', 0)
        full_proverb = result.get('full_proverb', '')
        difficulty_level = result.get('difficulty_level', 0)
        score = result.get('score', 0)
        confidence = result.get('confidence', 0.0)
        processing_time = result.get('processing_time', 0.0)
        
        # 속담 텍스트 길이 제한
        if len(full_proverb) > 27:
            proverb_display = full_proverb[:24] + "..."
        else:
            proverb_display = full_proverb
        
        # 난이도 이름
        level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(difficulty_level, {})
        level_name = level_info.get('name', f'{difficulty_level}단계')
        
        # 분석 세부 점수
        breakdown = result.get('analysis_breakdown', {})
        linguistic_score = breakdown.get('linguistic_analysis', {}).get('linguistic_score', 0)
        ai_score = breakdown.get('ai_analysis', {}).get('ai_score', 0)
        
        print(f"{proverb_id:<4} {proverb_display:<30} {level_name:<8} {score:<4} "
              f"{confidence:.1%:<8} {linguistic_score:.3f:<8} {ai_score:.3f:<8} {processing_time:.3f}s")
    
    # 요약 정보
    if max_rows and len(successful_results) > max_rows:
        print(f"... (총 {len(successful_results)}개 중 {max_rows}개 표시)")


def calculate_difficulty_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    📈 난이도 분석 결과의 통계를 계산합니다.
    
    Args:
        results: 분석 결과 목록
        
    Returns:
        Dict: 통계 정보
    """
    if not results:
        return {}
    
    successful_results = [r for r in results if r.get('difficulty_level', 0) > 0]
    
    if not successful_results:
        return {"total": len(results), "successful": 0}
    
    # 기본 통계
    total_count = len(results)
    successful_count = len(successful_results)
    success_rate = successful_count / total_count if total_count > 0 else 0
    
    # 점수 통계
    scores = [r.get('score', 0) for r in successful_results]
    total_score = sum(scores)
    average_score = total_score / successful_count if successful_count > 0 else 0
    
    # 신뢰도 통계
    confidences = [r.get('confidence', 0.0) for r in successful_results]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # 처리 시간 통계
    processing_times = [r.get('processing_time', 0.0) for r in successful_results]
    total_processing_time = sum(processing_times)
    average_processing_time = total_processing_time / len(processing_times) if processing_times else 0
    
    # 난이도별 분포
    difficulty_distribution = {1: 0, 2: 0, 3: 0}
    for result in successful_results:
        level = result.get('difficulty_level', 0)
        if level in difficulty_distribution:
            difficulty_distribution[level] += 1
    
    return {
        "total": total_count,
        "successful": successful_count,
        "success_rate": success_rate,
        "total_score": total_score,
        "average_score": average_score,
        "average_confidence": average_confidence,
        "total_processing_time": total_processing_time,
        "average_processing_time": average_processing_time,
        "difficulty_distribution": difficulty_distribution
    }


def print_statistics_summary(stats: Dict[str, Any]) -> None:
    """
    📊 통계 정보를 요약해서 출력합니다.
    
    Args:
        stats: 통계 정보
    """
    if not stats:
        print("❌ 표시할 통계 정보가 없습니다.")
        return
    
    print(f"\n📈 분석 통계 요약:")
    print(f"   - 총 속담: {stats.get('total', 0)}개")
    print(f"   - 성공 분석: {stats.get('successful', 0)}개 ({stats.get('success_rate', 0):.1%})")
    print(f"   - 총 점수: {stats.get('total_score', 0)}점")
    print(f"   - 평균 점수: {stats.get('average_score', 0):.2f}점")
    print(f"   - 평균 신뢰도: {stats.get('average_confidence', 0):.1%}")
    print(f"   - 평균 처리 시간: {stats.get('average_processing_time', 0):.3f}초")
    
    # 난이도별 분포
    distribution = stats.get('difficulty_distribution', {})
    if distribution:
        print(f"\n🎯 난이도별 분포:")
        for level, count in distribution.items():
            level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(level, {})
            level_name = level_info.get('name', f'{level}단계')
            percentage = (count / stats.get('successful', 1)) * 100
            print(f"   - {level_name}: {count}개 ({percentage:.1f}%)")


def format_analysis_result(result: Dict[str, Any], detailed: bool = False) -> str:
    """
    🎯 개별 분석 결과를 포맷팅합니다.
    
    Args:
        result: 분석 결과
        detailed: 상세 정보 포함 여부
        
    Returns:
        str: 포맷팅된 결과 문자열
    """
    if not result or result.get('difficulty_level', 0) <= 0:
        return "❌ 분석 실패"
    
    proverb = result.get('full_proverb', '')
    level = result.get('difficulty_level', 0)
    score = result.get('score', 0)
    confidence = result.get('confidence', 0.0)
    processing_time = result.get('processing_time', 0.0)
    
    level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(level, {})
    level_name = level_info.get('name', f'{level}단계')
    
    basic_info = f"'{proverb}' → {level_name} ({score}점, 신뢰도 {confidence:.1%})"
    
    if not detailed:
        return basic_info
    
    # 상세 정보 추가
    message = result.get('message', '')
    detailed_info = f"{basic_info}\n   💭 {message}\n   ⏱️ 처리시간: {processing_time:.3f}초"
    
    # 분석 세부사항
    breakdown = result.get('analysis_breakdown', {})
    if breakdown:
        if 'linguistic_analysis' in breakdown:
            ling = breakdown['linguistic_analysis']
            detailed_info += f"\n   📝 언어학적: {ling.get('linguistic_score', 0):.3f}"
        
        if 'ai_analysis' in breakdown:
            ai = breakdown['ai_analysis']
            detailed_info += f"\n   🤖 AI 모델: {ai.get('ai_score', 0):.3f}"
    
    return detailed_info


def get_difficulty_color(level: int) -> str:
    """
    🎨 난이도에 따른 색상 코드를 반환합니다.
    
    Args:
        level: 난이도 레벨 (1-3)
        
    Returns:
        str: ANSI 색상 코드
    """
    colors = {
        1: '\033[92m',  # 초록색 (쉬움)
        2: '\033[93m',  # 노란색 (보통)  
        3: '\033[91m',  # 빨간색 (어려움)
    }
    return colors.get(level, '\033[0m')  # 기본색


def print_colored_difficulty(level: int, text: str = None) -> None:
    """
    🌈 난이도에 따른 색상으로 텍스트를 출력합니다.
    
    Args:
        level: 난이도 레벨
        text: 출력할 텍스트 (None이면 기본 난이도명)
    """
    if text is None:
        level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(level, {})
        text = level_info.get('name', f'{level}단계')
    
    color = get_difficulty_color(level)
    reset = '\033[0m'
    print(f"{color}{text}{reset}")


def export_analysis_results(results: List[Dict[str, Any]], filepath: str = None) -> str:
    """
    💾 분석 결과를 파일로 내보냅니다.
    
    Args:
        results: 분석 결과 목록
        filepath: 저장할 파일 경로 (None이면 자동 생성)
        
    Returns:
        str: 저장된 파일 경로
    """
    if not results:
        raise ValueError("내보낼 분석 결과가 없습니다.")
    
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"difficulty_analysis_{timestamp}.txt"
    
    successful_results = [r for r in results if r.get('difficulty_level', 0) > 0]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("🎯 속담 게임 - 난이도 분석 결과\n")
        f.write("=" * 50 + "\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 분석: {len(results)}개\n")
        f.write(f"성공 분석: {len(successful_results)}개\n\n")
        
        # 결과 목록
        for i, result in enumerate(successful_results, 1):
            proverb = result.get('full_proverb', '')
            level = result.get('difficulty_level', 0)
            score = result.get('score', 0)
            confidence = result.get('confidence', 0.0)
            message = result.get('message', '')
            
            level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(level, {})
            level_name = level_info.get('name', f'{level}단계')
            
            f.write(f"{i:3d}. {proverb}\n")
            f.write(f"     → {level_name} ({score}점, 신뢰도 {confidence:.1%})\n")
            f.write(f"     💭 {message}\n\n")
        
        # 통계 정보
        stats = calculate_difficulty_statistics(results)
        f.write("\n📈 분석 통계:\n")
        f.write("-" * 30 + "\n")
        f.write(f"총 점수: {stats.get('total_score', 0)}점\n")
        f.write(f"평균 점수: {stats.get('average_score', 0):.2f}점\n")
        f.write(f"평균 신뢰도: {stats.get('average_confidence', 0):.1%}\n")
        
        # 난이도별 분포
        distribution = stats.get('difficulty_distribution', {})
        f.write(f"\n🎯 난이도별 분포:\n")
        for level, count in distribution.items():
            level_info = proverb_config.PROVERB_DIFFICULTY_LEVELS.get(level, {})
            level_name = level_info.get('name', f'{level}단계')
            percentage = (count / len(successful_results)) * 100 if successful_results else 0
            f.write(f"{level_name}: {count}개 ({percentage:.1f}%)\n")
    
    return filepath


if __name__ == "__main__":
    """
    🧪 유틸리티 함수 테스트
    """
    print("🧪 속담 게임 유틸리티 함수 테스트")
    
    # 테스트용 더미 데이터
    test_results = [
        {
            'proverb_id': 1,
            'full_proverb': '가는 말이 고와야 오는 말이 곱다',
            'difficulty_level': 1,
            'score': 100,
            'confidence': 0.85,
            'processing_time': 0.123,
            'message': '쉬움 난이도 (언어학적 분석 우세)'
        },
        {
            'proverb_id': 2, 
            'full_proverb': '사공이 많으면 배가 산으로 간다',
            'difficulty_level': 2,
            'score': 200,
            'confidence': 0.92,
            'processing_time': 0.156,
            'message': '보통 난이도 (복잡한 구조)'
        }
    ]
    
    # 테이블 출력 테스트
    print("\n📋 요약 테이블:")
    print_analysis_summary_table(test_results)
    
    print("\n📊 상세 테이블:")
    print_detailed_analysis_table(test_results)
    
    # 통계 계산 테스트
    stats = calculate_difficulty_statistics(test_results)
    print_statistics_summary(stats)
    
    print("\n✅ 유틸리티 함수 테스트 완료!")