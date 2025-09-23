// 속담 게임 - FastAPI 연동 JavaScript
$(function() {
  var $answerInput   = $('.answer-input');
  var $hintBtn       = $('.btn-hint');
  var $submitBtn     = $('.btn-submit');
  var $reloadBtn     = $('.btn-reload');
  var $currentScore  = $('#currentScore');
  var $timeFill      = $('#timeFill');
  var $timeDisplay   = $('#timeDisplay');
  var $correctCount  = $('#correctCount');
  var $streakCount   = $('#streakCount');
  var $questionText  = $('#questionText');
  var $hintDisplay   = $('#hintDisplay');
  var $hintContent   = $('#hintContent');
  
  // 모달 관련
  var $resultModal   = $('#resultModal');
  var $resultIcon    = $('#resultIcon');
  var $resultTitle   = $('#resultTitle');
  var $resultMessage = $('#resultMessage');
  var $gameOverModal = $('#gameOverModal');
  var $gameOverScoreText = $('#gameOverScoreText');
  var $gameOverMessage = $('#gameOverMessage');
  var $gameOverContinueBtn = $('#gameOverContinueBtn');

  // 게임 상태
  var gameId = null;
  var currentQuestion = null;
  var currentScore = 0;
  var correctCount = 0;
  var streakCount  = 0;
  var isGameActive = false;
  var hintShown = false;
  var wrongCount = 0;

  var totalTime     = 60; // 60초
  var remainingTime = totalTime;
  var timerInterval = null;
  var questionStartTime = null;
  var autoHintTimeout = null;

  // 게임 시작
  function startGame() {
    $.ajax({
      url: '/api/game/start',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({})
    }).done(function(resp) {
      if (resp.success) {
        gameId = resp.game_id;
        isGameActive = true;
        displayQuestion(resp.question);
        updateGameInfo(resp.game_info);
        startTimer();
        console.log('게임 시작:', resp);
      } else {
        showCustomAlert('❌ 오류', '게임을 시작할 수 없습니다.', 'error');
      }
    }).fail(function(xhr) {
      showCustomAlert('❌ 오류', '서버 연결 실패: ' + (xhr.responseJSON?.detail || xhr.statusText), 'error');
    });
  }

  // 문제 표시
  function displayQuestion(question) {
    currentQuestion = question;
    hintShown = false;
    wrongCount = 0;
    questionStartTime = Date.now();
    
    // 이전 자동 힌트 타이머 클리어
    if (autoHintTimeout) {
      clearTimeout(autoHintTimeout);
      autoHintTimeout = null;
    }
    
    var difficultyEmoji = {1: '🟢', 2: '🟡', 3: '🔴'};
    var difficultyText = difficultyEmoji[question.difficulty_level] || '🟡';
    
    $questionText.html(
      '<div class="difficulty-badge">' + difficultyText + ' ' + question.difficulty_name + '</div>' +
      '<div class="question-main">' + question.question_text + ' ______</div>'
    );
    
    // 힌트 영역 숨기기
    $hintDisplay.hide();
    
    $answerInput.val('').prop('disabled', false).show().focus();
    $submitBtn.show().prop('disabled', false).text('✅ 확인');
    $hintBtn.show().prop('disabled', false).text('💡 힌트');
    $reloadBtn.hide();
    
    // 10초 후 자동 힌트 타이머 설정
    autoHintTimeout = setTimeout(function() {
      if (!hintShown && currentQuestion && gameId) {
        showAutoHint();
      }
    }, 10000); // 10초
  }

  // 타이머 시작 (서버에서 받은 시간 기준)
  function startTimer() {
    clearInterval(timerInterval);
    updateTimeDisplay();
    
    timerInterval = setInterval(function() {
      if (!isGameActive) {
        clearInterval(timerInterval);
        return;
      }
      
      remainingTime--;
      updateTimeDisplay();
      
      if (remainingTime <= 0) {
        clearInterval(timerInterval);
        gameOver('시간 종료');
      }
    }, 1000);
  }

  // 게임 정보 업데이트
  function updateGameInfo(gameInfo) {
    if (gameInfo.remaining_time !== undefined) {
      remainingTime = gameInfo.remaining_time;
    }
    if (gameInfo.current_score !== undefined) {
      currentScore = gameInfo.current_score;
      $currentScore.text(currentScore.toLocaleString());
    }
    if (gameInfo.questions_answered !== undefined) {
      correctCount = gameInfo.questions_answered;
      $correctCount.text(correctCount);
    }
    if (gameInfo.streak_count !== undefined) {
      streakCount = gameInfo.streak_count;
      $streakCount.text(streakCount);
    }
  }

  function updateTimeDisplay() {
    var minutes = Math.floor(remainingTime / 60);
    var seconds = remainingTime % 60;
    var timeString = minutes + ':' + String(seconds).padStart(2, '0');
    $timeDisplay.text(timeString);

    var percentage = (remainingTime / totalTime) * 100;
    $timeFill.css('width', Math.max(0, percentage) + '%');

    $timeFill.removeClass('warning danger');
    $timeDisplay.removeClass('warning danger');
    if (percentage <= 20) {
      $timeFill.addClass('danger');
      $timeDisplay.addClass('danger');
    } else if (percentage <= 40) {
      $timeFill.addClass('warning');
      $timeDisplay.addClass('warning');
    }
  }

  function gameOver(reason) {
    isGameActive = false;
    clearInterval(timerInterval);
    
    // 자동 힌트 타이머 정리
    if (autoHintTimeout) {
      clearTimeout(autoHintTimeout);
      autoHintTimeout = null;
    }
    
    var message = '수고하셨습니다!';
    if (reason) message += ' (' + reason + ')';
    
    // 게임 종료 모달 표시
    showGameOverModal(currentScore, message, function() {
      // 랭킹 모달로 이동
      showRankModal(currentScore);
    });
    
    $timeDisplay.text("0:00");
    $answerInput.hide();
    $submitBtn.hide();
    $hintBtn.hide();
    $reloadBtn.show();
  }

  function showScoreAnimation(points) {
    $currentScore.addClass('score-pulse').css('color', '#28a745');
    setTimeout(function() {
      $currentScore.removeClass('score-pulse').css('color', '#667eea');
    }, 600);
    
    // 점수 팝업 효과
    var $popup = $('<div class="score-popup">+' + points + '점</div>');
    $popup.css({
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      fontSize: '24px',
      fontWeight: 'bold',
      color: '#28a745',
      zIndex: 1000,
      pointerEvents: 'none'
    });
    
    $('body').append($popup);
    $popup.animate({
      top: '30%',
      opacity: 0
    }, 1000, function() {
      $popup.remove();
    });
  }

  // 입력 효과
  $answerInput.on('focus', function() { 
    $(this).css('transform', 'translateY(-2px)'); 
  });
  $answerInput.on('blur', function() { 
    $(this).css('transform', 'translateY(0)'); 
  });

  // 힌트를 화면에 표시하는 함수
  function displayHint(hintText, isAuto = false) {
    hintShown = true;
    $hintContent.text(hintText);
    $hintDisplay.show();
    $hintBtn.prop('disabled', true).text('💡 힌트 표시됨');
    
    // 입력 필드로 포커스 이동
    setTimeout(function() {
      $answerInput.focus();
    }, 100);
  }

  // 10초 후 자동 힌트 표시
  function showAutoHint() {
    if (!gameId || !currentQuestion || hintShown) {
      return;
    }
    
    $.ajax({
      url: '/api/game/hint',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        game_id: gameId,
        question_id: currentQuestion.question_id
      })
    }).done(function(resp) {
      if (resp.success) {
        displayHint(resp.hint, true);
      }
    }).fail(function(xhr) {
      console.log('자동 힌트 요청 실패:', xhr);
    });
  }

  // 힌트 요청
  $hintBtn.on('click', function() {
    if (!gameId || !currentQuestion || hintShown) {
      return;
    }
    
    // 자동 힌트 타이머 취소 (수동으로 힌트 요청 시)
    if (autoHintTimeout) {
      clearTimeout(autoHintTimeout);
      autoHintTimeout = null;
    }
    
    $.ajax({
      url: '/api/game/hint',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        game_id: gameId,
        question_id: currentQuestion.question_id
      })
    }).done(function(resp) {
      if (resp.success) {
        displayHint(resp.hint, false);
      }
    }).fail(function(xhr) {
      showCustomAlert('❌ 오류', '힌트를 가져올 수 없습니다: ' + (xhr.responseJSON?.detail || xhr.statusText), 'error');
    });
  });

  // 정답 제출
  $submitBtn.on('click', function() {
    var answer = $.trim($answerInput.val());
    if (!answer) {
      showCustomAlert('⚠️ 알림', '답(속담 뒷부분)을 입력해주세요.', 'warning');
      $answerInput.focus();
      return;
    }
    
    if (!gameId || !currentQuestion) {
      showCustomAlert('❌ 오류', '게임 상태가 올바르지 않습니다.', 'error');
      return;
    }
    
    // 자동 힌트 타이머 정리 (정답 제출 시)
    if (autoHintTimeout) {
      clearTimeout(autoHintTimeout);
      autoHintTimeout = null;
    }
    
    // 버튼 비활성화
    $submitBtn.prop('disabled', true).text('확인 중...');
    $answerInput.prop('disabled', true);
    
    $.ajax({
      url: '/api/game/answer',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        answer: answer,
        game_id: gameId,
        question_id: currentQuestion.question_id
      })
    }).done(function(resp) {
      handleAnswerResponse(resp);
    }).fail(function(xhr) {
      showCustomAlert('❌ 오류', '서버 오류: ' + (xhr.responseJSON?.detail || xhr.statusText), 'error');
      // 버튼 재활성화
      $submitBtn.prop('disabled', false).text('✅ 확인');
      $answerInput.prop('disabled', false).focus();
    });
  });

  // 정답 응답 처리
  function handleAnswerResponse(resp) {
    if (resp.game_ended) {
      // 게임 종료
      isGameActive = false;
      if (resp.correct && resp.score_earned) {
        updateGameInfo({ current_score: currentScore + resp.score_earned });
        showScoreAnimation(resp.score_earned);
      }
      
      setTimeout(function() {
        gameOver(resp.message || '게임 종료');
      }, resp.correct ? 1000 : 0);
      return;
    }
    
    if (resp.correct) {
      // 정답 처리
      if (resp.score_earned) {
        showScoreAnimation(resp.score_earned);
      }
      updateGameInfo(resp.game_info);
      
      var message = resp.message + '\n유사도: ' + Math.round(resp.similarity * 100) + '%';
      showResultModal(true, '정답입니다!', '유사도: ' + Math.round(resp.similarity * 100) + '%', 1500);
      
      // 다음 문제로
      setTimeout(function() {
        if (resp.next_question) {
          displayQuestion(resp.next_question);
        } else {
          gameOver('모든 문제 완료');
        }
      }, 1000);
      
    } else {
      // 오답 처리
      wrongCount = resp.wrong_count || 0;
      updateGameInfo(resp.game_info);
      
      var message = resp.message + '\n유사도: ' + Math.round(resp.similarity * 100) + '%';
      
      if (resp.skip_to_next) {
        // 2회 오답으로 다음 문제로
        showResultModal(false, '2회 오답!', '정답: ' + (resp.correct_answer || ''), 2000);
        setTimeout(function() {
          if (resp.next_question) {
            displayQuestion(resp.next_question);
          } else {
            gameOver('게임 종료');
          }
        }, 2200);
      } else {
        // 1회 오답, 힌트 표시
        showResultModal(false, '오답입니다', '유사도: ' + Math.round(resp.similarity * 100) + '%', 1500);
        if (resp.show_hint && resp.hint) {
          setTimeout(function() {
            displayHint(resp.hint, true);
          }, 1600);
        }
        
        // 입력 필드 재활성화
        $answerInput.prop('disabled', false).select().focus();
        $submitBtn.prop('disabled', false).text('✅ 확인');
      }
    }
  }

  // Enter로 제출
  $answerInput.on('keypress', function(e) {
    if (e.which === 13 && !$submitBtn.prop('disabled')) { 
      $submitBtn.click(); 
    }
  });

  function showCustomAlert(title, message, type) {
    var emoji = type === 'success' ? '🎉' : (type === 'error' ? '❌' : (type === 'warning' ? '⚠️' : '💡'));
    alert(emoji + ' ' + title + '\n\n' + message);
  }

  // 결과 모달 표시
  function showResultModal(isCorrect, title, message, autoCloseMs = 2000) {
    $resultModal.removeClass('correct wrong');
    
    if (isCorrect) {
      $resultModal.addClass('correct');
      $resultIcon.text('🎉');
    } else {
      $resultModal.addClass('wrong');
      $resultIcon.text('❌');
    }
    
    $resultTitle.text(title);
    $resultMessage.text(message);
    
    $resultModal.css('display', 'flex').hide().fadeIn(200);
    
    // 자동으로 닫기
    setTimeout(function() {
      $resultModal.fadeOut(200);
    }, autoCloseMs);
  }

  // 게임 종료 모달 표시
  function showGameOverModal(score, message, callback) {
    $gameOverScoreText.text(score.toLocaleString());
    $gameOverMessage.text(message);
    $gameOverModal.css('display', 'flex').hide().fadeIn(200);
    
    // 확인 버튼 이벤트
    $gameOverContinueBtn.off('click').on('click', function() {
      $gameOverModal.fadeOut(200);
      if (callback) callback();
    });
  }

  // ===== 랭킹 모달 =====
  var $rankModal      = $('#rankModal');
  var $rankClose      = $('#rankCloseBtn');
  var $rankLater      = $('#rankLaterBtn');
  var $rankSave       = $('#rankSaveBtn');
  var $playerName     = $('#playerName');
  var $finalScoreText = $('#finalScoreText');
  var savingRank      = false;

  function showRankModal(finalScore) {
    $finalScoreText.text(finalScore.toLocaleString());
    $playerName.val('');
    $rankModal.css('display', 'flex').hide().fadeIn(160);
    setTimeout(function() { 
      $playerName.focus(); 
    }, 50);
  }
  
  function hideRankModal() {
    $rankModal.fadeOut(120);
  }
  
  $rankClose.on('click', hideRankModal);

  // 게임 재시작
  $reloadBtn.on('click', function() {
    location.reload();
  });

  // 메인 페이지로 이동
  $('.nav-title').on('click', function() {
    location.href = '/';
  });

  // 랭킹 저장
  $rankSave.on('click', function() {
    if (savingRank) return;
    
    var name = $.trim($playerName.val());
    if (!name) {
      alert('이름(또는 ID)을 입력해주세요.');
      $playerName.focus();
      return;
    }
    
    savingRank = true;
    $rankSave.prop('disabled', true).text('저장 중...');

    $.ajax({
      url: '/api/ranking/save',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        username: name,
        score: currentScore
      })
    }).done(function(resp) {
      if (resp.success) {
        alert('🏆 랭킹 저장 완료!\n' + resp.message);
        window.location.href = '/rankings';
      } else {
        alert('❌ 저장 실패: ' + resp.message);
        savingRank = false;
        $rankSave.prop('disabled', false).text('🏆 랭킹 등록');
      }
    }).fail(function(xhr) {
      alert('❌ 서버 저장 실패: ' + (xhr.responseJSON?.detail || xhr.statusText));
      savingRank = false;
      $rankSave.prop('disabled', false).text('🏆 랭킹 등록');
    });
  });

  // 페이지 로드 시 초기화
  $(document).ready(function() {
    console.log('속담 게임 페이지 로드');
    
    // 초기 UI 설정
    $questionText.html('<div class="welcome-message">🎯 게임을 시작할 준비가 되었습니다!</div>');
    $answerInput.hide();
    $submitBtn.hide();
    $hintBtn.hide();
    $reloadBtn.show().text('🎮 게임 시작');
    
    // 게임 시작 버튼 이벤트
    $reloadBtn.off('click').on('click', function() {
      $(this).text('게임 시작 중...').prop('disabled', true);
      startGame();
    });
  });
});
