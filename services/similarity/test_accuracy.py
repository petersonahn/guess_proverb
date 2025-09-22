#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
속담 유사도 검사 정확도 테스트 및 분석 도구

이 스크립트는 similarity_check.py의 정확도를 체계적으로 테스트하고
문제점을 분석하여 개선 방향을 제시합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from similarity_check import (
    get_proverbs_from_db, 
    get_threshold_by_length,
    generate_test_cases,
    check_proverb,
    enhanced_similarity_check
)
import random
import json
from collections import defaultdict

def analyze_test_case_accuracy():
    """
    각 속담에 대해 생성된 테스트 케이스들의 정확도를 분석합니다.
    """
    print("🔍 속담 유사도 검사 정확도 분석 시작")
    print("=" * 60)
    
    # 데이터베이스에서 속담 로드
    try:
        proverbs = get_proverbs_from_db()
        print(f"📚 총 {len(proverbs)}개 속담 로드 완료")
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return
    
    # 분석 결과 저장
    analysis_results = {
        "total_proverbs": len(proverbs),
        "accuracy_by_length": defaultdict(list),
        "false_positives": [],  # 잘못 인정된 답안들
        "false_negatives": [], # 정답인데 틀렸다고 나온 답안들
        "threshold_analysis": {},
        "test_case_performance": defaultdict(list)
    }
    
    # 각 속담별로 테스트 케이스 분석
    for i, (pid, question, answer, hint) in enumerate(proverbs[:20]):  # 처음 20개만 테스트
        print(f"\n📝 [{i+1}/20] 속담 분석: {question} → {answer}")
        
        # 현재 속담의 임계값
        threshold = get_threshold_by_length(answer)
        print(f"   임계값: {threshold:.2f} (길이: {len(answer)}글자)")
        
        # 테스트 케이스 생성
        test_cases = generate_test_cases(answer)
        print(f"   생성된 테스트 케이스: {len(test_cases)}개")
        
        # 각 테스트 케이스 분석
        answers = [row[2] for row in proverbs]  # 모든 답안 리스트
        
        correct_count = 0
        incorrect_count = 0
        case_results = []
        
        for test_input, description in test_cases:
            # 유사도 검사
            is_correct, matched_answer, similarity = check_proverb(test_input, answers, threshold)
            
            # 예상 결과 vs 실제 결과 비교
            should_be_correct = (description == "정확한 답안" or 
                               "띄어쓰기" in description or 
                               "조사 변경" in description)
            
            case_result = {
                "input": test_input,
                "description": description,
                "similarity": similarity,
                "is_correct": is_correct,
                "matched_answer": matched_answer,
                "should_be_correct": should_be_correct,
                "result_type": ""
            }
            
            # 결과 분류
            if is_correct and matched_answer == answer:
                if should_be_correct:
                    case_result["result_type"] = "TRUE_POSITIVE"  # 정답을 정답으로
                    correct_count += 1
                else:
                    case_result["result_type"] = "FALSE_POSITIVE"  # 오답을 정답으로
                    analysis_results["false_positives"].append({
                        "proverb": answer,
                        "input": test_input,
                        "description": description,
                        "similarity": similarity,
                        "threshold": threshold
                    })
            else:
                if should_be_correct:
                    case_result["result_type"] = "FALSE_NEGATIVE"  # 정답을 오답으로
                    analysis_results["false_negatives"].append({
                        "proverb": answer,
                        "input": test_input,
                        "description": description,
                        "similarity": similarity,
                        "threshold": threshold
                    })
                    incorrect_count += 1
                else:
                    case_result["result_type"] = "TRUE_NEGATIVE"  # 오답을 오답으로
                    correct_count += 1
            
            case_results.append(case_result)
        
        # 속담별 정확도 계산
        total_cases = len(test_cases)
        accuracy = correct_count / total_cases if total_cases > 0 else 0
        
        print(f"   정확도: {accuracy:.2f} ({correct_count}/{total_cases})")
        
        # 길이별 정확도 저장
        length_category = get_length_category(len(answer))
        analysis_results["accuracy_by_length"][length_category].append(accuracy)
        
        # 테스트 케이스 성능 저장
        analysis_results["test_case_performance"][answer] = case_results
        
        # 임계값 분석 저장
        analysis_results["threshold_analysis"][answer] = {
            "length": len(answer),
            "threshold": threshold,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_cases": total_cases
        }
    
    # 전체 분석 결과 출력
    print_analysis_summary(analysis_results)
    
    # 결과를 파일로 저장
    save_analysis_results(analysis_results)
    
    return analysis_results

def get_length_category(length):
    """속담 길이를 카테고리로 분류"""
    if length <= 5:
        return "매우짧음(≤5)"
    elif length <= 10:
        return "짧음(6-10)"
    elif length <= 15:
        return "중간(11-15)"
    elif length <= 20:
        return "긴편(16-20)"
    else:
        return "매우김(≥21)"

def print_analysis_summary(results):
    """분석 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("📊 정확도 분석 결과 요약")
    print("=" * 60)
    
    # 길이별 평균 정확도
    print("\n📏 길이별 평균 정확도:")
    for category, accuracies in results["accuracy_by_length"].items():
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"   {category}: {avg_accuracy:.3f} ({len(accuracies)}개 속담)")
    
    # False Positive 분석
    print(f"\n❌ False Positive (잘못 인정된 답안): {len(results['false_positives'])}개")
    if results["false_positives"]:
        print("   상위 5개:")
        for fp in results["false_positives"][:5]:
            print(f"   - '{fp['input']}' → '{fp['proverb']}' ({fp['description']})")
            print(f"     유사도: {fp['similarity']:.3f}, 임계값: {fp['threshold']:.3f}")
    
    # False Negative 분석
    print(f"\n⚠️ False Negative (정답인데 틀렸다고 나온 답안): {len(results['false_negatives'])}개")
    if results["false_negatives"]:
        print("   상위 5개:")
        for fn in results["false_negatives"][:5]:
            print(f"   - '{fn['input']}' → '{fn['proverb']}' ({fn['description']})")
            print(f"     유사도: {fn['similarity']:.3f}, 임계값: {fn['threshold']:.3f}")
    
    # 임계값 분석
    print(f"\n🎯 임계값 분석:")
    threshold_groups = defaultdict(list)
    for proverb, data in results["threshold_analysis"].items():
        threshold_groups[data["threshold"]].append(data["accuracy"])
    
    for threshold, accuracies in sorted(threshold_groups.items()):
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"   임계값 {threshold:.2f}: 평균 정확도 {avg_accuracy:.3f} ({len(accuracies)}개 속담)")

def save_analysis_results(results):
    """분석 결과를 JSON 파일로 저장"""
    output_file = "accuracy_analysis_results.json"
    
    # JSON 직렬화를 위해 defaultdict를 일반 dict로 변환
    json_results = {
        "total_proverbs": results["total_proverbs"],
        "accuracy_by_length": dict(results["accuracy_by_length"]),
        "false_positives": results["false_positives"],
        "false_negatives": results["false_negatives"],
        "threshold_analysis": results["threshold_analysis"],
        "test_case_performance": dict(results["test_case_performance"])
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 분석 결과가 '{output_file}'에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")

def suggest_improvements(results):
    """분석 결과를 바탕으로 개선 방안 제시"""
    print("\n" + "=" * 60)
    print("💡 개선 방안 제시")
    print("=" * 60)
    
    # False Positive 패턴 분석
    fp_patterns = defaultdict(int)
    for fp in results["false_positives"]:
        if "어미 변화" in fp["description"]:
            fp_patterns["어미 변화"] += 1
        elif "오타" in fp["description"]:
            fp_patterns["오타"] += 1
        elif "단어 순서" in fp["description"]:
            fp_patterns["단어 순서"] += 1
        else:
            fp_patterns["기타"] += 1
    
    print("🔍 False Positive 주요 패턴:")
    for pattern, count in sorted(fp_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {pattern}: {count}개")
    
    # False Negative 패턴 분석
    fn_patterns = defaultdict(int)
    for fn in results["false_negatives"]:
        if "띄어쓰기" in fn["description"]:
            fn_patterns["띄어쓰기"] += 1
        elif "조사 변경" in fn["description"]:
            fn_patterns["조사 변경"] += 1
        elif "정확한 답안" in fn["description"]:
            fn_patterns["정확한 답안"] += 1
        else:
            fn_patterns["기타"] += 1
    
    print("\n🔍 False Negative 주요 패턴:")
    for pattern, count in sorted(fn_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {pattern}: {count}개")
    
    # 개선 제안
    print("\n📝 구체적인 개선 제안:")
    
    if fp_patterns.get("어미 변화", 0) > 0:
        print("1. 어미 변화 패널티 강화 필요")
        print("   - apply_penalty()에서 어미 변화 패널티를 0.6 → 0.4로 강화")
    
    if fn_patterns.get("띄어쓰기", 0) > 0:
        print("2. 띄어쓰기 관련 임계값 조정 필요")
        print("   - 띄어쓰기만 다른 경우 더 관대하게 처리")
    
    if fn_patterns.get("정확한 답안", 0) > 0:
        print("3. 기본 임계값이 너무 높을 수 있음")
        print("   - 길이별 임계값 전반적으로 0.02-0.05 낮춤 검토")

def interactive_test():
    """대화형 테스트 모드"""
    print("\n🎮 대화형 테스트 모드")
    print("속담을 입력하면 테스트 케이스를 생성하고 정확도를 분석합니다.")
    print("종료하려면 'quit' 입력")
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    while True:
        answer = input("\n테스트할 속담 입력: ").strip()
        if answer.lower() == 'quit':
            break
        
        if not answer:
            continue
        
        print(f"\n🔍 '{answer}' 분석 중...")
        
        # 테스트 케이스 생성
        test_cases = generate_test_cases(answer)
        threshold = get_threshold_by_length(answer)
        
        print(f"임계값: {threshold:.2f}")
        print(f"생성된 테스트 케이스: {len(test_cases)}개")
        print("-" * 40)
        
        for i, (test_input, description) in enumerate(test_cases, 1):
            is_correct, matched_answer, similarity = check_proverb(test_input, answers, threshold)
            
            status = "✅" if is_correct and matched_answer == answer else "❌"
            print(f"{i:2d}. {status} {test_input}")
            print(f"     {description}")
            print(f"     유사도: {similarity:.3f} (임계값: {threshold:.2f})")
            
            if matched_answer != answer:
                print(f"     매칭된 답안: {matched_answer}")

if __name__ == "__main__":
    print("🎯 속담 유사도 검사 정확도 분석 도구")
    print("=" * 60)
    
    try:
        # 자동 분석 실행
        results = analyze_test_case_accuracy()
        
        # 개선 방안 제시
        suggest_improvements(results)
        
        # 대화형 테스트 제공
        while True:
            choice = input("\n대화형 테스트를 실행하시겠습니까? (y/n): ").strip().lower()
            if choice == 'y':
                interactive_test()
                break
            elif choice == 'n':
                print("분석 완료!")
                break
            else:
                print("y 또는 n을 입력해주세요.")
                
    except KeyboardInterrupt:
        print("\n\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
