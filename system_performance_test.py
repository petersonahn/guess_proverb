"""
속담 난이도 분석 시스템 성능 테스트

정확도 판별 기준과 각 구성 요소 점수를 상세히 분석합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.includes.analyzer import ProverbDifficultyAnalyzer

def system_performance_test():
    """시스템 성능 테스트"""
    print("🔬 속담 난이도 분석 시스템 성능 테스트")
    print("=" * 70)
    
    # 전문가가 레이블링한 기준 속담들
    test_cases = [
        # 레벨 1 (쉬움) - 일상적으로 자주 사용
        ("티끌 모아 태산", 1),
        ("내 코가 석 자", 1),
        ("가재는 게 편이라", 1),
        ("말이 씨가 된다", 1),
        
        # 레벨 2 (보통) - 교육적으로 알려짐
        ("등잔 밑이 어둡다", 2),
        ("백문이 불여일견", 2),
        ("개천에서 용 난다", 2),
        ("고생 끝에 낙이 온다", 2),
        ("원숭이도 나무에서 떨어진다", 2),
        ("사공이 많으면 배가 산으로 간다", 2),
        
        # 레벨 3 (어려움) - 고어 포함, 복잡한 구조
        ("가자니 태산이요 돌아서자니 숭산이라", 3),
        ("금강산도 식후경", 3),
        ("백지장도 맞들면 낫다", 3),
        ("낮말은 새가 듣고 밤말은 쥐가 듣는다", 3),
    ]
    
    try:
        analyzer = ProverbDifficultyAnalyzer()
        
        print("📊 정확도 판별 기준:")
        print("  🟢 레벨 1 (쉬움): 일상 사용 빈도 높음, 단순 구조")
        print("  🟡 레벨 2 (보통): 교육적으로 알려짐, 중간 복잡성")
        print("  🔴 레벨 3 (어려움): 고어 포함, 복잡 구조, 사용 빈도 낮음")
        print()
        
        print("🔍 상세 분석 결과:")
        print("-" * 70)
        print(f"{'속담':<30} {'예상':<4} {'실제':<4} {'사용빈도':<8} {'복잡성':<8} {'최종':<8} {'게임점수':<6} {'결과'}")
        print("-" * 70)
        
        correct = 0
        total = len(test_cases)
        level_stats = {1: [], 2: [], 3: []}
        
        for proverb, expected in test_cases:
            result = analyzer.calculate_final_difficulty(proverb)
            actual = result['difficulty_level']
            final_score = result['final_score']
            
            # 구성 요소 점수
            breakdown = result['breakdown']
            usage_freq = breakdown['usage_frequency_analysis']['usage_frequency_score']
            complexity = breakdown['linguistic_complexity_analysis']['complexity_score']
            
            # 게임에서 사용할 점수
            game_score = actual
            
            # 정확도 체크
            is_correct = actual == expected
            if is_correct:
                correct += 1
            
            level_stats[expected].append({
                'proverb': proverb,
                'actual': actual,
                'score': final_score,
                'correct': is_correct,
                'usage_freq': usage_freq,
                'complexity': complexity
            })
            
            # 결과 출력
            proverb_short = proverb[:25] + "..." if len(proverb) > 25 else proverb
            status = "✅" if is_correct else "❌"
            
            print(f"{proverb_short:<30} {expected:<4} {actual:<4} {usage_freq:<8.3f} {complexity:<8.3f} {final_score:<8.3f} {game_score:<6} {status}")
        
        # 전체 성능 요약
        accuracy = (correct / total) * 100
        print("\n" + "=" * 70)
        print(f"🎯 전체 정확도: {correct}/{total} ({accuracy:.1f}%)")
        
        # 레벨별 성능
        print(f"\n📊 레벨별 성능:")
        level_names = {1: "쉬움", 2: "보통", 3: "어려움"}
        
        for level in [1, 2, 3]:
            data = level_stats[level]
            if data:
                level_correct = sum(1 for d in data if d['correct'])
                level_total = len(data)
                level_acc = (level_correct / level_total) * 100
                
                avg_score = sum(d['score'] for d in data) / len(data)
                avg_usage = sum(d['usage_freq'] for d in data) / len(data)
                avg_complexity = sum(d['complexity'] for d in data) / len(data)
                
                print(f"  🎯 레벨 {level} ({level_names[level]}): {level_correct}/{level_total} ({level_acc:.1f}%)")
                print(f"     평균 점수: {avg_score:.3f} | 사용빈도: {avg_usage:.3f} | 복잡성: {avg_complexity:.3f}")
        
        # 오분류 분석
        print(f"\n🔍 오분류 분석:")
        misclassified = []
        for level in [1, 2, 3]:
            for data in level_stats[level]:
                if not data['correct']:
                    misclassified.append((data['proverb'], level, data['actual'], data['score']))
        
        if misclassified:
            for proverb, expected, actual, score in misclassified:
                direction = "과소평가" if actual < expected else "과대평가"
                print(f"  ❌ {direction}: '{proverb[:40]}' (예상:{expected} → 실제:{actual}, 점수:{score:.3f})")
        else:
            print("  ✅ 오분류 없음!")
        
        # 경계값 분석
        print(f"\n⚙️ 현재 시스템 설정:")
        print(f"  📏 경계값: 레벨1 ≤ 0.58 | 레벨2: 0.58~0.68 | 레벨3 > 0.68")
        print(f"  ⚖️ 가중치: 사용빈도 60% + 언어복잡성 40%")
        
        # 성능 개선 제안
        print(f"\n💡 성능 개선 제안:")
        
        if accuracy < 70:
            print(f"  ⚠️ 정확도 개선 필요 ({accuracy:.1f}% < 70%)")
            
            # 레벨별 문제 분석
            for level in [1, 2, 3]:
                data = level_stats[level]
                if data:
                    level_correct = sum(1 for d in data if d['correct'])
                    level_acc = (level_correct / len(data)) * 100
                    
                    if level_acc < 60:
                        print(f"    🔧 레벨 {level} ({level_names[level]}) 성능 부족: {level_acc:.1f}%")
                        
                        # 평균 점수 분석
                        wrong_items = [d for d in data if not d['correct']]
                        if wrong_items:
                            avg_usage = sum(d['usage_freq'] for d in wrong_items) / len(wrong_items)
                            avg_complexity = sum(d['complexity'] for d in wrong_items) / len(wrong_items)
                            
                            if level == 1 and avg_usage < 0.25:
                                print(f"       💡 사용빈도 점수 낮음({avg_usage:.3f}) → AI 패턴 문장 조정 필요")
                            elif level == 3 and avg_complexity < 0.4:
                                print(f"       💡 복잡성 점수 낮음({avg_complexity:.3f}) → 언어학적 분석 강화 필요")
        else:
            print(f"  ✅ 양호한 성능 ({accuracy:.1f}% ≥ 70%)")
            print(f"  🎮 게임 적용 가능한 수준")
        
        # 게임 점수 시스템 정보
        print(f"\n🎮 게임 점수 시스템:")
        game_total = sum(actual for _, expected in test_cases for actual in [analyzer.calculate_final_difficulty(_)['difficulty_level']])
        print(f"  📊 테스트 속담 총 게임 점수: {game_total}점")
        print(f"  🏆 점수 체계: 쉬움=1점, 보통=2점, 어려움=3점")
        
        # 전체 데이터베이스 예상 점수 (84개 속담 기준)
        print(f"  📈 전체 DB 예상 점수: 약 179점 (8×1 + 57×2 + 19×3)")
        
        analyzer.close()
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "level_stats": level_stats
        }
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 시스템 성능 테스트 시작\n")
    
    result = system_performance_test()
    
    if result:
        accuracy = result["accuracy"]
        print(f"\n" + "=" * 70)
        print(f"🎯 최종 성능 평가: {accuracy:.1f}%")
        
        if accuracy >= 80:
            print("🎉 우수! 게임 적용 준비 완료")
        elif accuracy >= 70:
            print("✅ 양호! 게임 적용 가능")
        elif accuracy >= 60:
            print("🔧 보통! 추가 개선 권장")
        else:
            print("⚠️ 부족! 시스템 재검토 필요")
            
        print("\n📝 이 결과를 바탕으로 시스템 개선을 진행할 수 있습니다.")
    else:
        print("❌ 성능 테스트 실패")
