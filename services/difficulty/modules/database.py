"""
🎯 속담 게임 - 데이터베이스 연결 및 관리 모듈

이 모듈은 MySQL 데이터베이스와의 연결을 관리하고
proverb 테이블에서 속담 데이터를 조회하는 기능을 제공합니다.

주요 기능:
1. MySQL 데이터베이스 연결 관리
2. proverb 테이블 데이터 조회
3. 속담 전체 텍스트 생성 (question + answer)
4. 연결 상태 모니터링

테이블 구조:
- proverb (id, question, answer, hint)
- 데이터베이스: proverb_game
- 사용자: root, 비밀번호: 0000
"""

import sys
import traceback
import logging
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

# config 모듈 import
try:
    from config import proverb_config
except ImportError as e:
    print(f"❌ config.py import 실패: {e}")
    sys.exit(1)

# 데이터베이스 라이브러리 import
try:
    import mysql.connector
    from mysql.connector import Error
    import pymysql
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError as e:
    print(f"❌ 데이터베이스 라이브러리 import 실패: {e}")
    print("해결 방법: pip install mysql-connector-python pymysql sqlalchemy")
    sys.exit(1)


class ProverbDatabase:
    """
    🗄️ 속담 게임 전용 데이터베이스 관리 클래스
    
    MySQL proverb_game 데이터베이스의 proverb 테이블과 연동하여
    속담 데이터를 조회하고 관리하는 기능을 제공합니다.
    
    특징:
    - 연결 풀링으로 성능 최적화
    - 자동 재연결 기능
    - 트랜잭션 안전성
    - 상세한 에러 처리
    """
    
    def __init__(self):
        """
        🔧 데이터베이스 연결 초기화
        
        config.py의 설정을 사용하여 MySQL 데이터베이스에 연결합니다.
        """
        self.config = proverb_config
        self.connection = None
        self.engine = None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        print(f"🗄️ 데이터베이스 연결 초기화 중...")
        print(f"📍 호스트: {self.config.DB_HOST}:{self.config.DB_PORT}")
        print(f"📊 데이터베이스: {self.config.DB_NAME}")
        print(f"👤 사용자: {self.config.DB_USER}")
        
        # 데이터베이스 연결 시도
        if not self._connect():
            raise Exception("데이터베이스 연결 실패")
        
        print(f"✅ 데이터베이스 연결 성공!")
    
    def _connect(self) -> bool:
        """
        🔌 데이터베이스 연결을 설정합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            # SQLAlchemy 엔진 생성
            connection_string = (
                f"mysql+pymysql://{self.config.DB_USER}:{self.config.DB_PASSWORD}"
                f"@{self.config.DB_HOST}:{self.config.DB_PORT}/{self.config.DB_NAME}"
                f"?charset=utf8mb4"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # 연결 테스트
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            # 직접 연결도 생성 (호환성용)
            self.connection = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                charset='utf8mb4',
                autocommit=True,
                pool_name='proverb_pool',
                pool_size=5
            )
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {str(e)}")
            self._print_connection_error_solutions(e)
            return False
    
    def _print_connection_error_solutions(self, error: Exception) -> None:
        """
        🔧 데이터베이스 연결 에러 해결 방법을 안내합니다.
        
        Args:
            error: 발생한 에러
        """
        print("\n🔧 데이터베이스 연결 문제 해결 방법:")
        
        error_str = str(error).lower()
        
        if "access denied" in error_str:
            print("🔒 인증 실패:")
            print("  1. 사용자명/비밀번호 확인 (root/0000)")
            print("  2. MySQL 서버 실행 상태 확인")
            print("  3. 사용자 권한 확인")
            
        elif "can't connect" in error_str or "connection refused" in error_str:
            print("🌐 연결 실패:")
            print("  1. MySQL 서버 실행 여부 확인")
            print("  2. 호스트/포트 정보 확인 (localhost:3306)")
            print("  3. 방화벽 설정 확인")
            
        elif "unknown database" in error_str:
            print("🗄️ 데이터베이스 없음:")
            print("  1. proverb_game 데이터베이스 생성")
            print("  2. CREATE DATABASE proverb_game;")
            
        else:
            print("📋 일반적인 해결 방법:")
            print("  1. MySQL 서버 재시작")
            print("  2. 네트워크 연결 상태 확인")
            print("  3. 데이터베이스 설정 재확인")
    
    def test_connection(self) -> bool:
        """
        🧪 데이터베이스 연결 상태를 테스트합니다.
        
        Returns:
            bool: 연결 상태
        """
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                return True
            return False
            
        except Exception as e:
            print(f"⚠️ 연결 테스트 실패: {str(e)}")
            return False
    
    def get_proverb_by_id(self, proverb_id: int) -> Optional[Dict[str, Any]]:
        """
        🔍 ID로 특정 속담을 조회합니다.
        
        Args:
            proverb_id: 조회할 속담 ID
            
        Returns:
            Optional[Dict]: 속담 데이터 또는 None
            {
                'id': 1,
                'question': '가는 말이 고와야',
                'answer': '오는 말이 곱다',
                'hint': '말의 예의에 관한 속담',
                'full_proverb': '가는 말이 고와야 오는 말이 곱다'
            }
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT id, question, answer, hint 
                    FROM proverb 
                    WHERE id = :proverb_id
                """)
                
                result = conn.execute(query, {"proverb_id": proverb_id})
                row = result.fetchone()
                
                if row:
                    proverb_data = {
                        'id': row[0],
                        'question': row[1],
                        'answer': row[2],
                        'hint': row[3],
                        'full_proverb': self._combine_proverb_parts(row[1], row[2])
                    }
                    return proverb_data
                
                return None
                
        except Exception as e:
            print(f"❌ 속담 조회 실패 (ID: {proverb_id}): {str(e)}")
            return None
    
    def get_all_proverbs(self) -> List[Dict[str, Any]]:
        """
        📚 데이터베이스의 모든 속담을 조회합니다.
        
        Returns:
            List[Dict]: 모든 속담 데이터 목록
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT id, question, answer, hint 
                    FROM proverb 
                    ORDER BY id
                """)
                
                result = conn.execute(query)
                rows = result.fetchall()
                
                proverbs = []
                for row in rows:
                    proverb_data = {
                        'id': row[0],
                        'question': row[1],
                        'answer': row[2],
                        'hint': row[3],
                        'full_proverb': self._combine_proverb_parts(row[1], row[2])
                    }
                    proverbs.append(proverb_data)
                
                print(f"📚 총 {len(proverbs)}개 속담 조회 완료")
                return proverbs
                
        except Exception as e:
            print(f"❌ 전체 속담 조회 실패: {str(e)}")
            return []
    
    def get_proverbs_batch(self, offset: int = 0, limit: int = 16) -> List[Dict[str, Any]]:
        """
        📦 속담을 배치 단위로 조회합니다.
        
        Args:
            offset: 시작 위치
            limit: 조회할 개수
            
        Returns:
            List[Dict]: 배치 속담 데이터 목록
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT id, question, answer, hint 
                    FROM proverb 
                    ORDER BY id
                    LIMIT :limit OFFSET :offset
                """)
                
                result = conn.execute(query, {"limit": limit, "offset": offset})
                rows = result.fetchall()
                
                proverbs = []
                for row in rows:
                    proverb_data = {
                        'id': row[0],
                        'question': row[1],
                        'answer': row[2],
                        'hint': row[3],
                        'full_proverb': self._combine_proverb_parts(row[1], row[2])
                    }
                    proverbs.append(proverb_data)
                
                return proverbs
                
        except Exception as e:
            print(f"❌ 배치 속담 조회 실패: {str(e)}")
            return []
    
    def get_proverb_count(self) -> int:
        """
        📊 데이터베이스의 총 속담 개수를 조회합니다.
        
        Returns:
            int: 총 속담 개수
        """
        try:
            with self.engine.connect() as conn:
                query = text("SELECT COUNT(*) FROM proverb")
                result = conn.execute(query)
                count = result.fetchone()[0]
                return count
                
        except Exception as e:
            print(f"❌ 속담 개수 조회 실패: {str(e)}")
            return 0
    
    def _combine_proverb_parts(self, question: str, answer: str) -> str:
        """
        🔗 속담의 앞부분과 뒷부분을 합칩니다.
        
        Args:
            question: 속담 앞부분
            answer: 속담 뒷부분
            
        Returns:
            str: 완성된 속담
        """
        if not question or not answer:
            return ""
        
        # 공백 정리
        question = question.strip()
        answer = answer.strip()
        
        # 적절한 연결
        if question.endswith(('은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로')):
            return f"{question} {answer}"
        else:
            return f"{question} {answer}"
    
    def check_table_exists(self) -> bool:
        """
        🔍 proverb 테이블의 존재 여부를 확인합니다.
        
        Returns:
            bool: 테이블 존재 여부
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = :db_name 
                    AND table_name = 'proverb'
                """)
                
                result = conn.execute(query, {"db_name": self.config.DB_NAME})
                count = result.fetchone()[0]
                return count > 0
                
        except Exception as e:
            print(f"❌ 테이블 존재 확인 실패: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        📋 데이터베이스 정보를 조회합니다.
        
        Returns:
            Dict: 데이터베이스 정보
        """
        try:
            info = {
                "host": self.config.DB_HOST,
                "port": self.config.DB_PORT,
                "database": self.config.DB_NAME,
                "user": self.config.DB_USER,
                "connection_status": self.test_connection(),
                "table_exists": self.check_table_exists(),
                "proverb_count": self.get_proverb_count() if self.check_table_exists() else 0
            }
            
            return info
            
        except Exception as e:
            print(f"❌ 데이터베이스 정보 조회 실패: {str(e)}")
            return {}
    
    @contextmanager
    def get_connection(self):
        """
        🔗 컨텍스트 매니저로 안전한 연결 사용
        
        사용 예시:
            with db.get_connection() as conn:
                # 데이터베이스 작업
                pass
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            print(f"❌ 연결 컨텍스트 에러: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def close(self):
        """
        🔒 데이터베이스 연결을 안전하게 종료합니다.
        """
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
                print("✅ MySQL 직접 연결 종료")
            
            if self.engine:
                self.engine.dispose()
                print("✅ SQLAlchemy 엔진 종료")
                
        except Exception as e:
            print(f"⚠️ 연결 종료 중 오류: {str(e)}")
    
    # def __del__(self):
    #     """소멸자에서 연결 정리"""
    #     self.close()


def test_database_connection():
    """
    🧪 데이터베이스 연결 및 기본 기능을 테스트합니다.
    """
    print("🧪 데이터베이스 연결 테스트 시작")
    print("=" * 50)
    
    try:
        # 데이터베이스 연결 생성
        db = ProverbDatabase()
        
        # 연결 상태 확인
        print(f"\n🔍 연결 상태 테스트:")
        is_connected = db.test_connection()
        print(f"연결 상태: {'✅ 성공' if is_connected else '❌ 실패'}")
        
        if not is_connected:
            return False
        
        # 데이터베이스 정보 조회
        print(f"\n📋 데이터베이스 정보:")
        db_info = db.get_database_info()
        for key, value in db_info.items():
            print(f"  - {key}: {value}")
        
        # 테이블 존재 확인
        table_exists = db.check_table_exists()
        print(f"\n📊 proverb 테이블: {'✅ 존재' if table_exists else '❌ 없음'}")
        
        if table_exists:
            # 속담 개수 확인
            count = db.get_proverb_count()
            print(f"📚 총 속담 개수: {count}개")
            
            if count > 0:
                # 첫 번째 속담 조회 테스트
                print(f"\n🔍 첫 번째 속담 조회:")
                first_proverb = db.get_proverb_by_id(1)
                if first_proverb:
                    print(f"  ID: {first_proverb['id']}")
                    print(f"  문제: {first_proverb['question']}")
                    print(f"  정답: {first_proverb['answer']}")
                    print(f"  힌트: {first_proverb['hint']}")
                    print(f"  전체: {first_proverb['full_proverb']}")
                
                # 배치 조회 테스트
                print(f"\n📦 배치 조회 테스트 (처음 3개):")
                batch_proverbs = db.get_proverbs_batch(0, 3)
                for proverb in batch_proverbs:
                    print(f"  [{proverb['id']}] {proverb['full_proverb']}")
        
        # 연결 종료
        db.close()
        
        print(f"\n✅ 데이터베이스 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 테스트 실패: {str(e)}")
        print(f"📋 에러 상세: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    """
    🚀 스크립트 직접 실행 시 테스트 함수 호출
    
    실행 방법:
        python database.py
    """
    print("🗄️ 속담 게임 - 데이터베이스 연결 테스트")
    print("MySQL proverb_game 데이터베이스 연결")
    print("사용자: root, 비밀번호: 0000")
    print()
    
    success = test_database_connection()
    sys.exit(0 if success else 1)
