#!/usr/bin/env python3
"""
🎯 속담 게임 - 점수 확인 스크립트

데이터베이스에 저장된 속담들의 난이도 분석 결과와 점수를 확인합니다.
캐싱된 결과를 우선 사용하여 빠르게 점수를 조회할 수 있습니다.

사용법:
    python check_scores.py                    # 전체 속담 점수 확인
    python check_scores.py --id 1            # 특정 ID 속담 점수 확인
    python check_scores.py --batch 10        # 처음 10개 속담 점수 확인
"""

import sys
import os
import argparse
from typing import List, Dict, Any

# 현재 파일의 부모 디렉토리들을 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from app.includes.analyzer import ProverbDifficultyAnalyzer
    from app.includes.utils import print_analysis_summary_table
except ImportError as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("해결 방법: 프로젝트 루트에서 실행하세요.")
    sys.exit(1)


def check_single_proverb_score(proverb_id: int) -> Dict[str, Any]:
    """
    🔍 개별 속담의 점수를 확인합니다.
    
    Args:
        proverb_id (int): 속담 ID
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    print(f"🔍 속담 ID {proverb_id} 점수 확인 중...")
    
    try:
        analyzer = ProverbDifficultyAnalyzer()
        result = analyzer.analyze_proverb_difficulty(proverb_id)
        analyzer.close()
        
        if result['difficulty_level'] > 0:
            print(f"\n✅ 분석 완료:")
            print(f"   📋 속담: {result['full_proverb']}")
            print(f"   🎯 난이도: {result['difficulty_level']}단계")
            print(f"   🏆 점수: {result['score']}점")
            print(f"   📊 신뢰도: {result['confidence']:.1%}")
            print(f"   ⏱️ 처리시간: {result['processing_time']:.3f}초")
            
            if "캐시됨" in result['message']:
                print(f"   💾 캐시: 사용됨 (빠른 조회)")
            else:
                print(f"   💾 캐시: 새로 분석됨")
        else:
            print(f"❌ 속담을 찾을 수 없습니다: {result['message']}")
            
        return result
        
    except Exception as e:
        print(f"❌ 점수 확인 실패: {str(e)}")
        return {}


def check_batch_scores(limit: int = 10) -> List[Dict[str, Any]]:
    """
    📦 여러 속담의 점수를 배치로 확인합니다.
    
    Args:
        limit (int): 확인할 속담 개수
        
    Returns:
        List[Dict[str, Any]]: 분석 결과 목록
    """
    print(f"📦 처음 {limit}개 속담 점수 확인 중...")
    
    try:
        analyzer = ProverbDifficultyAnalyzer()
        
        # 배치로 속담 조회
        batch_proverbs = analyzer.db.get_proverbs_batch(0, limit)
        results = []
        
        for proverb_data in batch_proverbs:
            result = analyzer.analyze_proverb_difficulty(proverb_id=proverb_data['id'])
            results.append(result)
        
        analyzer.close()
        
        # 결과 테이블 출력
        print(f"\n📋 점수 확인 결과 (총 {len(results)}개):")
        print_analysis_summary_table(results, max_rows=limit)
        
        # 점수 통계
        successful_results = [r for r in results if r['difficulty_level'] > 0]
        total_score = sum(r['score'] for r in successful_results)
        
        print(f"\n📊 점수 통계:")
        print(f"   - 성공 분석: {len(successful_results)}/{len(results)}개")
        print(f"   - 총 점수: {total_score}점")
        print(f"   - 평균 점수: {total_score/len(successful_results):.2f}점")
        
        return results
        
    except Exception as e:
        print(f"❌ 배치 점수 확인 실패: {str(e)}")
        return []


def check_all_scores() -> List[Dict[str, Any]]:
    """
    🌟 데이터베이스의 모든 속담 점수를 확인합니다.
    """
    print(f"🌟 모든 속담 점수 확인 중...")
    
    try:
        analyzer = ProverbDifficultyAnalyzer()
        
        # 전체 배치 분석 실행
        print("⏳ 전체 속담 분석 중... (시간이 걸릴 수 있습니다)")
        all_results = analyzer.batch_analyze_all_proverbs()
        
        analyzer.close()
        
        # 결과 요약
        successful_results = [r for r in all_results if r['difficulty_level'] > 0]
        total_score = sum(r['score'] for r in successful_results)
        
        print(f"\n🎉 전체 분석 완료!")
        print(f"   - 총 속담: {len(all_results)}개")
        print(f"   - 성공 분석: {len(successful_results)}개")
        print(f"   - 총 점수: {total_score}점")
        print(f"   - 평균 점수: {total_score/len(successful_results):.2f}점")
        
        # 난이도별 분포
        level_counts = {}
        for result in successful_results:
            level = result['difficulty_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print(f"\n📊 난이도별 분포:")
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            percentage = (count / len(successful_results)) * 100
            print(f"   - {level}단계 ({level}점): {count}개 ({percentage:.1f}%)")
        
        return all_results
        
    except Exception as e:
        print(f"❌ 전체 점수 확인 실패: {str(e)}")
        return []


def main():
    """
    🚀 메인 실행 함수
    """
    parser = argparse.ArgumentParser(
        description="속담 게임 점수 확인 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python check_scores.py                    # 전체 속담 점수 확인
  python check_scores.py --id 1            # ID 1번 속담 점수 확인
  python check_scores.py --batch 10        # 처음 10개 속담 점수 확인
        """
    )
    
    parser.add_argument('--id', type=int, help='확인할 속담 ID')
    parser.add_argument('--batch', type=int, help='배치로 확인할 속담 개수')
    
    args = parser.parse_args()
    
    print("🎯 속담 게임 - 점수 확인 도구")
    print("=" * 50)
    
    if args.id:
        # 개별 속담 점수 확인
        check_single_proverb_score(args.id)
        
    elif args.batch:
        # 배치 점수 확인
        check_batch_scores(args.batch)
        
    else:
        # 전체 점수 확인
        check_all_scores()
    
    print("\n✅ 점수 확인 완료!")


if __name__ == "__main__":
    main()
