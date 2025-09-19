#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
속담 유사도 검사 빠른 테스트 및 개선 도구
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

def quick_accuracy_test():
    """대표적인 속담들로 빠른 정확도 테스트"""
    print("🚀 빠른 정확도 테스트")
    print("=" * 50)
    
    # 대표 속담들 (길이별로 선별)
    test_proverbs = [
        ("내 코가", "석 자다", "짧은 속담"),
        ("목마른 놈이", "우물 판다", "중간 속담"),
        ("가는 말이 고와야", "오는 말이 곱다", "중간 속담"),
        ("물에 빠진 놈 건져놓으니", "보따리 내놓으라 한다", "긴 속담"),
        ("얌전한 고양이가", "부뚜막에 먼저 올라간다", "긴 속담")
    ]
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    total_issues = 0
    
    for question, answer, category in test_proverbs:
        print(f"\n📝 {category}: {question} → {answer}")
        
        threshold = get_threshold_by_length(answer)
        print(f"   임계값: {threshold:.2f} (길이: {len(answer)}글자)")
        
        # 핵심 테스트 케이스만 생성
        key_test_cases = [
            (answer, "정확한 답안"),
            (answer.replace(" ", ""), "띄어쓰기 제거"),
            (answer + " ", "끝 공백 추가"),
            (answer[:-1] + ("다" if not answer.endswith("다") else "라"), "어미 변화"),
        ]
        
        print("   핵심 테스트:")
        issues_found = 0
        
        for test_input, description in key_test_cases:
            is_correct, matched_answer, similarity = check_proverb(test_input, answers, threshold)
            
            # 예상 결과
            should_pass = description in ["정확한 답안", "띄어쓰기 제거", "끝 공백 추가"]
            
            if should_pass and (not is_correct or matched_answer != answer):
                print(f"   ❌ FAIL: '{test_input}' ({description})")
                print(f"       유사도: {similarity:.3f}, 매칭: {matched_answer}")
                issues_found += 1
                total_issues += 1
            elif not should_pass and is_correct and matched_answer == answer:
                print(f"   ⚠️  LOOSE: '{test_input}' ({description})")
                print(f"       유사도: {similarity:.3f} - 너무 관대할 수 있음")
            else:
                print(f"   ✅ PASS: '{test_input}' ({description})")
                print(f"       유사도: {similarity:.3f}")
        
        print(f"   문제점: {issues_found}개")
    
    print(f"\n📊 전체 요약:")
    print(f"   총 문제점: {total_issues}개")
    print(f"   테스트 속담: {len(test_proverbs)}개")
    
    return total_issues

def test_specific_issues():
    """특정 문제 상황들을 테스트"""
    print("\n🔍 특정 문제 상황 테스트")
    print("=" * 50)
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    # 문제가 될 수 있는 케이스들
    problem_cases = [
        # 띄어쓰기 문제
        ("석 자다", "석자다", "띄어쓰기 차이"),
        ("우물 판다", "우물판다", "띄어쓰기 차이"),
        
        # 조사 변경
        ("게 편이라", "게 편이다", "조사 변경"),
        ("개구리", "개구리가", "조사 추가"),
        
        # 어미 변화
        ("곱다", "고와", "어미 변화"),
        ("간다", "가네", "어미 변화"),
        
        # 숫자 표현
        ("석 자다", "3자다", "숫자 표현"),
        ("천 리 간다", "1000리 간다", "숫자 표현"),
    ]
    
    for correct_answer, test_input, issue_type in problem_cases:
        if correct_answer in answers:  # 실제 답안인 경우만 테스트
            threshold = get_threshold_by_length(correct_answer)
            is_correct, matched_answer, similarity = check_proverb(test_input, answers, threshold)
            
            # 결과 분석
            should_pass = issue_type in ["띄어쓰기 차이", "조사 변경", "숫자 표현"]
            
            status = "✅" if (should_pass and is_correct and matched_answer == correct_answer) else "❌"
            print(f"{status} {issue_type}: '{test_input}' → '{correct_answer}'")
            print(f"   유사도: {similarity:.3f}, 임계값: {threshold:.2f}")
            
            if matched_answer != correct_answer:
                print(f"   실제 매칭: {matched_answer}")

def suggest_threshold_adjustments():
    """임계값 조정 제안"""
    print("\n💡 임계값 조정 제안")
    print("=" * 50)
    
    proverbs = get_proverbs_from_db()
    
    # 길이별 그룹화
    length_groups = {}
    for _, _, answer, _ in proverbs:
        length = len(answer)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(answer)
    
    print("현재 임계값 분포:")
    for length in sorted(length_groups.keys()):
        threshold = get_threshold_by_length("x" * length)  # 길이만큼 더미 문자열 생성
        count = len(length_groups[length])
        print(f"   {length}글자: {threshold:.2f} ({count}개 속담)")
        
        # 예시 속담
        if count > 0:
            example = length_groups[length][0]
            print(f"      예시: {example}")

def interactive_fix_test():
    """대화형 수정 테스트"""
    print("\n🛠️  대화형 수정 테스트")
    print("속담과 사용자 입력을 넣으면 현재 시스템의 판단을 보여줍니다.")
    print("종료하려면 'quit' 입력")
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    while True:
        print("\n" + "-" * 40)
        correct_answer = input("정답 속담: ").strip()
        if correct_answer.lower() == 'quit':
            break
        
        user_input = input("사용자 입력: ").strip()
        if not user_input:
            continue
        
        threshold = get_threshold_by_length(correct_answer)
        is_correct, matched_answer, similarity = check_proverb(user_input, answers, threshold)
        
        print(f"\n결과:")
        print(f"   정답: {correct_answer}")
        print(f"   입력: {user_input}")
        print(f"   임계값: {threshold:.2f}")
        print(f"   유사도: {similarity:.3f}")
        print(f"   판정: {'✅ 정답' if is_correct and matched_answer == correct_answer else '❌ 오답'}")
        
        if matched_answer != correct_answer:
            print(f"   매칭된 답안: {matched_answer}")
        
        # 개선 제안
        if not is_correct or matched_answer != correct_answer:
            if similarity >= threshold * 0.9:
                print("   💡 임계값을 약간 낮추면 통과할 수 있습니다.")
            elif "다" in user_input and "다" in correct_answer:
                print("   💡 어미 변화 패널티가 과도할 수 있습니다.")

if __name__ == "__main__":
    print("🎯 속담 유사도 검사 빠른 테스트 도구")
    print("=" * 60)
    
    try:
        # 1. 빠른 정확도 테스트
        total_issues = quick_accuracy_test()
        
        # 2. 특정 문제 상황 테스트
        test_specific_issues()
        
        # 3. 임계값 분석
        suggest_threshold_adjustments()
        
        # 4. 대화형 테스트
        print(f"\n총 {total_issues}개의 문제점이 발견되었습니다.")
        
        choice = input("\n대화형 수정 테스트를 실행하시겠습니까? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_fix_test()
        
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
