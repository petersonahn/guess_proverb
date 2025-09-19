#!/usr/bin/env python3
"""
🎯 속담 게임 실행 스크립트

FastAPI 기반의 속담 게임 서버를 실행합니다.
- 난이도 측정 기능 (ProverbDifficultyAnalyzer)
- 정답 유사도 측정 기능 (similarity_check.py)
- 60초 제한시간 게임
- 점수 시스템 (쉬움: 100점, 보통: 200점, 어려움: 300점)
- 힌트 시스템
- 랭킹 시스템

실행 방법:
    python run_game.py
    또는
    conda activate aa
    python run_game.py

서버 주소: http://localhost:8001
"""

import sys
import os
import subprocess
import time

def check_conda_environment():
    """conda 가상환경 확인"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'aa':
        print("⚠️ conda 가상환경 'aa'가 활성화되지 않았습니다.")
        print("다음 명령어로 가상환경을 활성화한 후 다시 실행해주세요:")
        print("conda activate aa")
        return False
    return True

def check_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인"""
    # 패키지명과 import명이 다른 경우를 매핑
    package_import_map = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'mysql-connector-python': 'mysql.connector',
        'sentence-transformers': 'sentence_transformers',
        'torch': 'torch',
        'transformers': 'transformers',
        'jinja2': 'jinja2'
    }
    
    missing_packages = []
    for package_name, import_name in package_import_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 설치해주세요:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_database_connection():
    """데이터베이스 연결 확인"""
    try:
        import mysql.connector
        from app.core.config import proverb_config
        
        conn = mysql.connector.connect(
            host=proverb_config.DB_HOST,
            port=proverb_config.DB_PORT,
            user=proverb_config.DB_USER,
            password=proverb_config.DB_PASSWORD,
            database=proverb_config.DB_NAME,
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        
        # 테이블 존재 확인
        cursor.execute("SHOW TABLES LIKE 'proverb'")
        proverb_table = cursor.fetchone()
        
        cursor.execute("SHOW TABLES LIKE 'user'")
        user_table = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not proverb_table:
            print("❌ 'proverb' 테이블이 존재하지 않습니다.")
            print("database/proverb.sql 파일을 실행하여 테이블을 생성해주세요.")
            return False
        
        if not user_table:
            print("❌ 'user' 테이블이 존재하지 않습니다.")
            print("database/user.sql 파일을 실행하여 테이블을 생성해주세요.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {str(e)}")
        print("MySQL 서버가 실행 중인지 확인하고, 데이터베이스 설정을 확인해주세요.")
        print("설정 파일: app/core/config.py")
        return False

def run_server():
    """서버 실행"""
    try:
        print("🚀 속담 게임 서버를 시작합니다... (최적화 버전)")
        print("=" * 60)
        print("🌐 서버 주소: http://127.0.0.1:8080")
        print("📖 API 문서: http://127.0.0.1:8080/docs")
        print("🎮 게임 플레이: http://127.0.0.1:8080/")
        print("🏆 랭킹 보기: http://127.0.0.1:8080/rankings")
        print("💡 AI 모델은 첫 게임 시작 시 로드됩니다")
        print("=" * 60)
        print("서버를 중지하려면 Ctrl+C를 누르세요.")
        print()
        
        # 서버 실행 (app 디렉토리에서 실행)
        os.chdir('app')
        subprocess.run([
            sys.executable, 'main.py'
        ])
        
    except KeyboardInterrupt:
        print("\n✅ 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 실패: {str(e)}")

def main():
    """메인 함수"""
    print("🎯 속담 게임 - 서버 시작 스크립트")
    print("=" * 60)
    
    # 1. conda 환경 확인
    print("1. conda 가상환경 확인 중...")
    if not check_conda_environment():
        return
    print("✅ conda 가상환경 'aa' 활성화됨")
    
    # 2. 패키지 설치 확인
    print("\n2. 필수 패키지 설치 확인 중...")
    if not check_dependencies():
        return
    print("✅ 모든 필수 패키지가 설치되어 있습니다")
    
    # 3. 데이터베이스 연결 확인
    print("\n3. 데이터베이스 연결 확인 중...")
    if not check_database_connection():
        return
    print("✅ 데이터베이스 연결 성공")
    
    # 4. 서버 실행
    print("\n4. 서버 실행 중...")
    time.sleep(1)
    run_server()

if __name__ == "__main__":
    main()
