"""
속담 난이도 분석 시스템 성능 분석 도구

- 정확도 판별 기준 분석
- 각 구성 요소별 점수 분석  
- 성능 개선 방향 제시
- 게임용 최종 결과 확인
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.includes.analyzer import ProverbDifficultyAnalyzer
import json
from datetime import datetime

def performance_analysis():
    """종합적인 성능 분석"""
    print("🔬 속담 난이도 분석 시스템 성능 분석")
    print("=" * 80)
    print(f"📅 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 정확도 판별을 위한 기준 속담들 (전문가 레이블)
    expert_labeled_proverbs = [
        # 🟢 레벨 1 (쉬움) - 일상적으로 자주 사용되고 누구나 아는 속담
        ("티끌 모아 태산", 1, "매우 유명하고 짧으며 일상적으로 자주 인용됨"),
        ("내 코가 석 자", 1, "일상 대화에서 자주 사용되는 표현"),
        ("가재는 게 편이라", 1, "단순한 구조와 친숙한 어휘"),
        ("말이 씨가 된다", 1, "현대적 표현으로 이해하기 쉬움"),
        ("우물 안 개구리", 1, "교육과정에서 자주 접하는 표현"),
        
        # 🟡 레벨 2 (보통) - 교육적으로 알려져 있지만 일상에서는 덜 사용
        ("등잔 밑이 어둡다", 2, "교육적으로 유명하지만 현대적 사용 빈도는 중간"),
        ("백문이 불여일견", 2, "한자어 포함, 교육적 맥락에서 주로 사용"),
        ("개천에서 용 난다", 2, "은유적 표현, 중간 길이"),
        ("고생 끝에 낙이 온다", 2, "격려용으로 사용되지만 구어체에서는 덜 사용"),
        ("원숭이도 나무에서 떨어진다", 2, "중간 길이, 교훈적 내용"),
        ("사공이 많으면 배가 산으로 간다", 2, "길고 복잡한 구조"),
        ("호랑이도 제 말 하면 온다", 2, "조건문 구조, 중간 복잡성"),
        
        # 🔴 레벨 3 (어려움) - 고어 포함, 복잡한 구조, 잘 알려지지 않음
        ("가자니 태산이요 돌아서자니 숭산이라", 3, "고어 종결어미, 매우 긴 구조, 현대에 거의 사용 안됨"),
        ("금강산도 식후경", 3, "한자어 + 고어, 현대적 사용 빈도 낮음"),
        ("백지장도 맞들면 낫다", 3, "고어 표현 '맞들면', 현대에 잘 안 쓰임"),
        ("낮말은 새가 듣고 밤말은 쥐가 듣는다", 3, "매우 긴 구조, 복잡한 대조 표현"),
        ("뱁새 황새 따라가다 다리 찢어진다", 3, "복잡한 은유, 긴 구조"),
    ]
    
    try:
        analyzer = ProverbDifficultyAnalyzer()
        
        print("📊 정확도 판별 기준:")
        print("  🟢 레벨 1 (쉬움): 일상적 사용 빈도 높음, 단순한 구조, 현대적 어휘")
        print("  🟡 레벨 2 (보통): 교육적으로 알려짐, 중간 복잡성, 은유적 표현")
        print("  🔴 레벨 3 (어려움): 고어 포함, 복잡한 구조, 현대적 사용 빈도 낮음")
        print()
        
        # 상세 분석 결과 저장
        detailed_results = []
        
        # 성능 통계
        correct_count = 0
        total_count = len(expert_labeled_proverbs)
        level_stats = {1: {"correct": 0, "total": 0, "scores": []}, 
                      2: {"correct": 0, "total": 0, "scores": []}, 
                      3: {"correct": 0, "total": 0, "scores": []}}
        
        print("🔍 상세 분석 결과:")
        print("=" * 80)
        header = f"{'속담':<35} {'예상':<4} {'실제':<4} {'사용빈도':<8} {'복잡성':<8} {'최종점수':<8} {'게임점수':<6} {'결과'}"
        print(header)
        print("-" * 80)
        
        for proverb_text, expected_level, reason in expert_labeled_proverbs:
            # 상세 분석
            analysis_result = analyzer.calculate_final_difficulty(proverb_text)
            actual_level = analysis_result['difficulty_level']
            confidence = analysis_result['confidence']
            final_score = analysis_result['final_score']
            
            # 구성 요소 점수 추출
            breakdown = analysis_result['breakdown']
            usage_freq_score = breakdown['usage_frequency_analysis']['usage_frequency_score']
            complexity_score = breakdown['linguistic_complexity_analysis']['complexity_score']
            
            # AI 분석 상세 정보
            ai_analysis = breakdown['usage_frequency_analysis']
            
            # 게임에서 사용할 점수 (레벨 * 1점)
            game_score = actual_level
            
            # 정확도 판정
            is_correct = actual_level == expected_level
            if is_correct:
                correct_count += 1
            
            # 통계 업데이트
            level_stats[expected_level]["total"] += 1
            level_stats[expected_level]["scores"].append(final_score)
            if is_correct:
                level_stats[expected_level]["correct"] += 1
            
            # 결과 출력
            proverb_short = proverb_text[:30] + "..." if len(proverb_text) > 30 else proverb_text
            status = "✅" if is_correct else "❌"
            
            print(f"{proverb_short:<35} {expected_level:<4} {actual_level:<4} {usage_freq_score:<8.3f} {complexity_score:<8.3f} {final_score:<8.3f} {game_score:<6} {status}")
            
            # 상세 결과 저장
            detailed_result = {
                "proverb": proverb_text,
                "expected_level": expected_level,
                "actual_level": actual_level,
                "is_correct": is_correct,
                "confidence": confidence,
                "final_score": final_score,
                "game_score": game_score,
                "reason": reason,
                "usage_frequency_analysis": {
                    "overall_score": usage_freq_score,
                    "daily_context_similarity": ai_analysis["daily_context_similarity"],
                    "educational_similarity": ai_analysis["educational_similarity"], 
                    "media_similarity": ai_analysis["media_similarity"],
                    "overall_familiarity": ai_analysis["overall_familiarity"],
                    "structure_familiarity": ai_analysis["structure_familiarity"]
                },
                "linguistic_complexity_analysis": {
                    "overall_score": complexity_score,
                    "linguistic_complexity": breakdown['linguistic_complexity_analysis']['linguistic_complexity'],
                    "vocabulary_difficulty": breakdown['linguistic_complexity_analysis']['vocabulary_difficulty'],
                    "structural_simplicity": breakdown['linguistic_complexity_analysis']['structural_simplicity']
                }
            }
            detailed_results.append(detailed_result)
        
        # 전체 성능 분석
        accuracy = (correct_count / total_count) * 100
        print("\n" + "=" * 80)
        print("📈 전체 성능 분석")
        print("=" * 80)
        print(f"🎯 전체 정확도: {correct_count}/{total_count} ({accuracy:.1f}%)")
        
        # 레벨별 성능 분석
        print(f"\n📊 레벨별 성능:")
        level_names = {1: "쉬움", 2: "보통", 3: "어려움"}
        for level in [1, 2, 3]:
            correct = level_stats[level]["correct"]
            total = level_stats[level]["total"]
            level_acc = (correct / total * 100) if total > 0 else 0
            
            if level_stats[level]["scores"]:
                avg_score = sum(level_stats[level]["scores"]) / len(level_stats[level]["scores"])
                min_score = min(level_stats[level]["scores"])
                max_score = max(level_stats[level]["scores"])
                print(f"  🎯 레벨 {level} ({level_names[level]}): {correct}/{total} ({level_acc:.1f}%) | 평균점수: {avg_score:.3f} | 범위: {min_score:.3f}-{max_score:.3f}")
        
        # 오분류 분석
        print(f"\n🔍 오분류 분석:")
        misclassified = [r for r in detailed_results if not r["is_correct"]]
        if misclassified:
            for result in misclassified:
                expected = result["expected_level"]
                actual = result["actual_level"]
                score = result["final_score"]
                proverb = result["proverb"][:40] + "..." if len(result["proverb"]) > 40 else result["proverb"]
                
                if actual < expected:
                    print(f"  ⬇️ 과소평가: '{proverb}' (예상: {expected} → 실제: {actual}, 점수: {score:.3f})")
                else:
                    print(f"  ⬆️ 과대평가: '{proverb}' (예상: {expected} → 실제: {actual}, 점수: {score:.3f})")
        else:
            print("  ✅ 오분류 없음!")
        
        # 경계값 분석
        print(f"\n⚙️ 현재 경계값 분석:")
        print(f"  레벨 1 (쉬움): 점수 ≤ 0.58")
        print(f"  레벨 2 (보통): 0.58 < 점수 ≤ 0.68")
        print(f"  레벨 3 (어려움): 점수 > 0.68")
        
        # 성능 개선 제안
        print(f"\n💡 성능 개선 제안:")
        
        if accuracy < 70:
            print(f"  🔧 정확도 개선 필요 ({accuracy:.1f}% < 70%)")
            
            # 레벨별 문제점 분석
            for level in [1, 2, 3]:
                level_acc = (level_stats[level]["correct"] / level_stats[level]["total"] * 100) if level_stats[level]["total"] > 0 else 0
                if level_acc < 60:
                    print(f"    ⚠️ 레벨 {level} ({level_names[level]}) 정확도 낮음: {level_acc:.1f}%")
                    
                    # 해당 레벨 오분류 원인 분석
                    level_misclassified = [r for r in misclassified if r["expected_level"] == level]
                    if level_misclassified:
                        avg_usage_freq = sum(r["usage_frequency_analysis"]["overall_score"] for r in level_misclassified) / len(level_misclassified)
                        avg_complexity = sum(r["linguistic_complexity_analysis"]["overall_score"] for r in level_misclassified) / len(level_misclassified)
                        
                        print(f"      📊 오분류 평균 - 사용빈도: {avg_usage_freq:.3f}, 복잡성: {avg_complexity:.3f}")
                        
                        if level == 1 and avg_usage_freq < 0.3:
                            print(f"      💡 제안: 사용 빈도 점수가 낮음. AI 패턴 문장을 더 일상적으로 조정 필요")
                        elif level == 3 and avg_complexity < 0.4:
                            print(f"      💡 제안: 복잡성 점수가 낮음. 언어학적 복잡성 가중치 조정 필요")
        
        # 경계값 조정 제안
        level_1_scores = [r["final_score"] for r in detailed_results if r["expected_level"] == 1]
        level_2_scores = [r["final_score"] for r in detailed_results if r["expected_level"] == 2]  
        level_3_scores = [r["final_score"] for r in detailed_results if r["expected_level"] == 3]
        
        if level_1_scores and level_2_scores:
            level_1_max = max(level_1_scores)
            level_2_min = min(level_2_scores)
            if level_1_max > 0.58:
                new_boundary_1_2 = (level_1_max + level_2_min) / 2
                print(f"    🎯 레벨 1-2 경계값 조정 제안: 0.58 → {new_boundary_1_2:.3f}")
        
        if level_2_scores and level_3_scores:
            level_2_max = max(level_2_scores)
            level_3_min = min(level_3_scores)
            if level_2_max > 0.68:
                new_boundary_2_3 = (level_2_max + level_3_min) / 2
                print(f"    🎯 레벨 2-3 경계값 조정 제안: 0.68 → {new_boundary_2_3:.3f}")
        
        # 가중치 조정 제안
        usage_freq_weight = 0.6
        complexity_weight = 0.4
        
        # 사용 빈도 vs 복잡성 효과성 분석
        usage_freq_correct = sum(1 for r in detailed_results if r["is_correct"] and 
                                abs(r["usage_frequency_analysis"]["overall_score"] - (1 - r["expected_level"] / 3)) < 0.2)
        complexity_correct = sum(1 for r in detailed_results if r["is_correct"] and 
                               abs(r["linguistic_complexity_analysis"]["overall_score"] - (r["expected_level"] / 3)) < 0.2)
        
        if usage_freq_correct < complexity_correct:
            print(f"    ⚖️ 가중치 조정 제안: 언어학적 복잡성 비중 증가 (현재: 사용빈도 60% vs 복잡성 40%)")
        elif usage_freq_correct > complexity_correct:
            print(f"    ⚖️ 가중치 조정 제안: 사용 빈도 비중 증가 (현재: 사용빈도 60% vs 복잡성 40%)")
        
        if accuracy >= 70:
            print(f"  ✅ 양호한 성능 ({accuracy:.1f}% ≥ 70%)")
            print(f"    🎮 게임 적용 가능한 수준")
        
        # 게임 점수 시스템 분석
        total_game_score = sum(r["game_score"] for r in detailed_results)
        avg_game_score = total_game_score / len(detailed_results)
        
        print(f"\n🎮 게임 점수 시스템 분석:")
        print(f"  📊 테스트 속담 총 점수: {total_game_score}점")
        print(f"  📊 평균 게임 점수: {avg_game_score:.1f}점")
        print(f"  🎯 점수 배분: 레벨1={level_stats[1]['total']}개×1점, 레벨2={level_stats[2]['total']}개×2점, 레벨3={level_stats[3]['total']}개×3점")
        
        # 결과 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"performance_analysis_results_{timestamp}.json"
        
        analysis_summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": accuracy,
            "total_proverbs": total_count,
            "correct_predictions": correct_count,
            "level_statistics": level_stats,
            "detailed_results": detailed_results,
            "current_thresholds": {
                "level_1_max": 0.58,
                "level_2_max": 0.68
            },
            "weights": {
                "usage_frequency": 0.6,
                "linguistic_complexity": 0.4
            }
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 결과 저장: {result_file}")
        
        analyzer.close()
        
        return analysis_summary
        
    except Exception as e:
        print(f"❌ 분석 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 속담 난이도 분석 성능 테스트 시작")
    result = performance_analysis()
    
    if result:
        accuracy = result["overall_accuracy"]
        print(f"\n🎯 최종 성능: {accuracy:.1f}%")
        
        if accuracy >= 80:
            print("🎉 우수한 성능! 게임 적용 준비 완료")
        elif accuracy >= 70:
            print("✅ 양호한 성능! 게임 적용 가능")
        elif accuracy >= 60:
            print("🔧 개선 필요! 추가 조정 권장")
        else:
            print("⚠️ 성능 부족! 시스템 재검토 필요")
    else:
        print("❌ 성능 분석 실패")
