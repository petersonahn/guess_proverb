-- 유저 테이블 생성
CREATE TABLE IF NOT EXISTS user (
    user_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '유저 고유 ID',
    username VARCHAR(50) NOT NULL COMMENT '유저명',
    total_score INT DEFAULT 0 COMMENT '점수(합산)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '등록 날짜'
) ENGINE=InnoDB 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci 
COMMENT='유저 게임 결과 저장 테이블';