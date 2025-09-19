"""
🧪 시스템 통합 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_config():
    """Config 모듈 테스트"""
    try:
        from app.core.config import proverb_config
        print("✅ Config 모듈 정상 로딩")
        print(f"   - 모델: {proverb_config.MODEL_NAME}")
        print(f"   - 디바이스: {proverb_config.DEVICE}")
        print(f"   - 데이터베이스: {proverb_config.DB_NAME}")
        return True
    except Exception as e:
        print(f"❌ Config 모듈 실패: {e}")
        return False

def test_database():
    """데이터베이스 연결 테스트"""
    try:
        from app.includes.dbconn import ProverbDatabase
        print("✅ Database 모듈 정상 로딩")
        
        # 연결 테스트
        db = ProverbDatabase()
        count = db.get_proverb_count()
        print(f"   - 속담 개수: {count}개")
        db.close()
        return True
    except Exception as e:
        print(f"❌ Database 연결 실패: {e}")
        return False

def test_utils():
    """유틸리티 모듈 테스트"""
    try:
        from app.includes.utils import validate_database_connection
        print("✅ Utils 모듈 정상 로딩")
        
        result = validate_database_connection()
        print(f"   - 연결 상태: {result['connected']}")
        return True
    except Exception as e:
        print(f"❌ Utils 모듈 실패: {e}")
        return False

def test_difficulty_analyzer():
    """난이도 분석기 테스트"""
    try:
        from app.includes.analyzer import ProverbDifficultyAnalyzer
        print("✅ Difficulty Analyzer 모듈 정상 로딩")
        
        # 분석기 초기화 (시간이 오래 걸림)
        print("⏳ AI 모델 로딩 중... (시간이 걸릴 수 있습니다)")
        analyzer = ProverbDifficultyAnalyzer()
        
        # 간단한 분석 테스트
        result = analyzer.analyze_proverb_difficulty(proverb_id=1)
        if result['difficulty_level'] > 0:
            print(f"   - 분석 성공: {result['full_proverb']} -> {result['difficulty_level']}단계")
        
        analyzer.close()
        return True
    except Exception as e:
        print(f"❌ Difficulty Analyzer 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 속담 게임 시스템 통합 테스트")
    print("=" * 50)
    
    tests = [
        ("Config 모듈", test_config),
        ("Database 연결", test_database),
        ("Utils 모듈", test_utils),
        ("Difficulty Analyzer", test_difficulty_analyzer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name} 테스트:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print(f"\n📊 테스트 결과 요약:")
    print("-" * 30)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{status} - {test_name}")
        if success:
            success_count += 1
    
    print(f"\n전체 결과: {success_count}/{len(results)} 테스트 통과")
    
    if success_count == len(results):
        print("🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 문제를 해결해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
