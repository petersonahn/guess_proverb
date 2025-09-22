#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 실제 게임 시나리오 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from similarity_check import (
    get_proverbs_from_db, 
    get_threshold_by_length,
    check_proverb
)

def test_real_scenarios():
    """실제 게임에서 발생할 수 있는 시나리오들을 테스트"""
    print("🎮 실제 게임 시나리오 테스트")
    print("=" * 50)
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    # 실제 사용자가 입력할 수 있는 패턴들
    test_scenarios = [
        # 띄어쓰기 관련
        ("석 자다", "석자다", "띄어쓰기 제거"),
        ("우물 판다", "우물판다", "띄어쓰기 제거"),
        ("게 편이라", "게편이라", "띄어쓰기 제거"),
        
        # 숫자 표현
        ("석 자다", "3자다", "숫자 변환"),
        ("석 자다", "3 자다", "숫자 변환"),
        ("천 리 간다", "1000리 간다", "숫자 변환"),
        ("세 살 버릇", "3살 버릇", "숫자 변환"),
        
        # 조사 변경
        ("게 편이라", "게 편이다", "조사 변경"),
        ("개구리", "개구리가", "조사 추가"),
        ("별 따기", "별따기", "띄어쓰기 제거"),
        
        # 어미 변화 (허용하면 안 되는 것들)
        ("곱다", "고와", "어미 변화 - 거부해야 함"),
        ("간다", "가네", "어미 변화 - 거부해야 함"),
        
        # 오타
        ("석 자다", "석 자디", "오타"),
        ("우물 판다", "우물 팬다", "오타"),
    ]
    
    print("\n📊 테스트 결과:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for correct_answer, user_input, test_type in test_scenarios:
        if correct_answer in answers:  # 실제 답안인 경우만
            threshold = get_threshold_by_length(correct_answer)
            is_correct, matched_answer, similarity = check_proverb(user_input, answers, threshold)
            
            # 예상 결과
            should_pass = test_type in [
                "띄어쓰기 제거", "숫자 변환", "조사 변경", "조사 추가"
            ]
            
            # 결과 판정
            actual_pass = is_correct and matched_answer == correct_answer
            
            if should_pass == actual_pass:
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
                failed += 1
            
            print(f"{status} [{test_type}]")
            print(f"   정답: '{correct_answer}' → 입력: '{user_input}'")
            print(f"   유사도: {similarity:.3f}, 임계값: {threshold:.2f}")
            print(f"   결과: {'정답' if actual_pass else '오답'} (예상: {'정답' if should_pass else '오답'})")
            
            if matched_answer != correct_answer:
                print(f"   매칭된 답안: {matched_answer}")
            print()
    
    print("=" * 60)
    print(f"📈 전체 결과: {passed}개 성공, {failed}개 실패")
    print(f"정확도: {passed/(passed+failed)*100:.1f}%")
    
    return passed, failed

def run_actual_game_test():
    """실제 게임을 짧게 실행해서 정확도 체감"""
    print("\n🎯 실제 게임 체감 테스트")
    print("몇 개의 속담을 직접 테스트해보세요")
    print("=" * 50)
    
    proverbs = get_proverbs_from_db()
    answers = [row[2] for row in proverbs]
    
    # 대표적인 속담들
    test_proverbs = [
        (1, "가는 말이 고와야", "오는 말이 곱다", "말의 왕래와 관련된 속담입니다."),
        (3, "개천에서", "용 난다", "보잘것없는 곳에서도 훌륭한 인물이 나올 수 있다는 뜻입니다."),
        (6, "내 코가", "석 자다", "자신의 일도 제대로 해결하지 못하는데 남을 도울 수 없다는 의미입니다."),
        (10, "목마른 놈이", "우물 판다", "급한 사람이 직접 해결책을 찾는다는 의미입니다."),
    ]
    
    for pid, question, answer, hint in test_proverbs:
        print(f"\n📝 문제: {question}")
        print(f"힌트: {hint}")
        
        threshold = get_threshold_by_length(answer)
        
        # 다양한 입력 시도
        test_inputs = [
            answer,  # 정확한 답
            answer.replace(" ", ""),  # 띄어쓰기 제거
            answer + "요",  # 어미 변화
        ]
        
        # 숫자 변환 테스트
        if "석 자다" in answer:
            test_inputs.append("3자다")
            test_inputs.append("3 자다")
        
        print(f"임계값: {threshold:.2f}")
        print("테스트 입력들:")
        
        for test_input in test_inputs:
            is_correct, matched_answer, similarity = check_proverb(test_input, answers, threshold)
            
            status = "✅" if is_correct and matched_answer == answer else "❌"
            print(f"  {status} '{test_input}' → 유사도: {similarity:.3f}")
            
            if matched_answer != answer:
                print(f"      (매칭: {matched_answer})")

if __name__ == "__main__":
    print("🎯 속담 유사도 검사 최종 테스트")
    print("=" * 60)
    
    try:
        # 1. 시나리오 테스트
        passed, failed = test_real_scenarios()
        
        # 2. 실제 게임 체감 테스트
        run_actual_game_test()
        
        print(f"\n🏆 최종 결과:")
        print(f"시나리오 테스트 정확도: {passed/(passed+failed)*100:.1f}%")
        
        if passed >= failed * 2:  # 성공이 실패의 2배 이상
            print("✅ 전체적으로 좋은 성능입니다!")
        else:
            print("⚠️ 추가 개선이 필요합니다.")
            
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
