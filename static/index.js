// jQuery + API (DB questions, embedding scoring, user-table ranking)
$(function() {
  var $answerInput   = $('.answer-input');
  var $hintBtn       = $('.btn-hint');
  var $submitBtn     = $('.btn-submit');
  var $reloadBtn  = $('.btn-reload');
  var $currentScore  = $('#currentScore');
  var $timeFill      = $('#timeFill');
  var $timeDisplay   = $('#timeDisplay');
  var $correctCount  = $('#correctCount');
  var $streakCount   = $('#streakCount');
  var $questionText  = $('#questionText');

  var questions    = [];
  var currentScore = 0;
  var correctCount = 0;
  var streakCount  = 0;
  var currentIndex = 0;

  var totalTime     = 59; // 1분
  var remainingTime = totalTime;
  var timerInterval = null;

  function loadQuestion(idx) {
    if (!questions.length) return;
    currentIndex = idx;
    $questionText.text(questions[currentIndex].question + ' ______');
    $reloadBtn.hide();
    $answerInput.show();
    $submitBtn.show();
    $hintBtn.show();
    $answerInput.val('').prop('disabled', false);
    $answerInput.focus();
  }

  function nextQuestion() {
    var next = (currentIndex + 1) % questions.length;
    loadQuestion(next);
  }

  function startTimer() {
    clearInterval(timerInterval);
    updateTimeDisplay(); // 초기 표시 보정
    timerInterval = setInterval(function() {
      remainingTime--;
      updateTimeDisplay();
      if (remainingTime <= -1) {
        clearInterval(timerInterval);
        gameOver();
      }
    }, 1000);
  }

  function updateTimeDisplay() {
    var minutes = Math.floor(remainingTime/60);
    var seconds = remainingTime%60;
    var timeString = minutes + ':' + String(seconds).padStart(2,'0');
    $timeDisplay.text(timeString);

    var percentage = (remainingTime/totalTime)*100;
    $timeFill.css('width', percentage + '%');

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

  function gameOver() {
    showCustomAlert('⏰ 시간 종료!', '게임이 끝났습니다!\n총 점수: ' + currentScore.toLocaleString() + '점\n정답 개수: ' + correctCount + '개', 'info');
    showRankModal(currentScore);
    $timeDisplay.text("0:00");
    $answerInput.hide();
    $submitBtn.hide();
    $hintBtn.hide();
    $reloadBtn.show();
  }

  function updateScore(points) {
    currentScore += points;
    correctCount += 1;
    streakCount  += 1;

    $currentScore.text(currentScore.toLocaleString());
    $correctCount.text(String(correctCount));
    $streakCount.text(String(streakCount));

    $currentScore.addClass('score-pulse').css('color','#28a745');
    setTimeout(function(){
      $currentScore.removeClass('score-pulse').css('color','#667eea');
    },600);
  }

  function resetStreak() {
    streakCount = 0;
    $streakCount.text('0');
  }

  // 입력 효과
  $answerInput.on('focus', function(){ $(this).css('transform','translateY(-2px)'); });
  $answerInput.on('blur',  function(){ $(this).css('transform','translateY(0)');   });

  // 힌트
  $hintBtn.on('click', function(){
    var hint = (questions[currentIndex] && questions[currentIndex].hint) || '이 문제에는 힌트가 없습니다.';
    showCustomAlert('💡 힌트', hint, 'info');
  });

  // 제출
  $submitBtn.on('click', function(){
    var answer = $answerInput.val();
    if ($.trim(answer) === '') {
      showCustomAlert('⚠️ 알림','답(속담 뒷부분)을 입력해주세요.','warning');
      $answerInput.focus();
      return;
    }
    var q = questions[currentIndex];
    $.ajax({
      url: '/api/score',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ id: q.id, answer: answer })
    }).done(function(resp){
      var pts = Number(resp.awarded) || 0;
      if (resp.correct) {
        updateScore(pts);
        showCustomAlert('🎉 정답 처리!', '유사도: ' + Math.round(resp.similarity) + '%  (+' + pts + '점)', 'success');
        setTimeout(function(){ nextQuestion(); }, 700);
      } else {
        showCustomAlert('❌ 조금 더!', '유사도: ' + Math.round(resp.similarity) + '% — 조금 더 정확히 입력해보세요.', 'error');
        resetStreak();
        $answerInput.select();
      }
    }).fail(function(xhr){
      alert('채점 에러: ' + (xhr.responseJSON && xhr.responseJSON.detail || xhr.statusText));
    });
  });

  // Enter로 제출
  $answerInput.on('keypress', function(e){
    if (e.which === 13) { $submitBtn.click(); }
  });

  function showCustomAlert(title, message, type) {
    var emoji = type === 'success' ? '🎉' : (type === 'error' ? '❌' : (type === 'warning' ? '⚠️' : '💡'));
    alert(emoji + ' ' + title + '\n\n' + message);
  }
  function createSuccessParticles(){ console.log('🎉 Success particles!'); }

  // ===== 랭킹 모달 (user 테이블 사용) =====
  var $rankModal      = $('#rankModal');
  var $rankClose      = $('#rankCloseBtn');
  var $rankLater      = $('#rankLaterBtn');
  var $rankSave       = $('#rankSaveBtn');
  var $playerName     = $('#playerName');
  var $finalScoreText = $('#finalScoreText');
  var savingRank      = false;

  function showRankModal(finalScore){
    $finalScoreText.text(finalScore.toLocaleString());
    $playerName.val('');
    $rankModal.css('display','flex').hide().fadeIn(160);
    setTimeout(function(){ $playerName.focus(); }, 50);
  }
  function hideRankModal(){
    $rankModal.fadeOut(120);
  }
  $rankClose.on('click', hideRankModal);
  $rankLater.on('click', function(){ hideRankModal(); });

  $reloadBtn.on('click', function(){
    location.reload();
  });
  $rankSave.on('click', function(){
    if (savingRank) return;
    var name = $.trim($playerName.val());
    if (!name){
      alert('이름(또는 ID)을 입력해주세요.');
      $playerName.focus();
      return;
    }
    savingRank = true;
    $rankSave.prop('disabled', true).text('저장 중...');

    $.ajax({
      url: '/api/users',            // ✅ user 랭킹 API
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({
        username: name,             // ✅ 필드명 변경
        total_score: currentScore   // ✅ 필드명 변경
      })
    }).done(function(){
      window.location.href = '/rankings?just=1';
    }).fail(function(xhr){
      alert('서버 저장 실패: ' + (xhr.responseJSON && xhr.responseJSON.detail || xhr.statusText));
      savingRank = false;
      $rankSave.prop('disabled', false).text('🏆 랭킹 등록');
    });
  });

  // 문제 로드
  function fetchQuestions() {
    return $.getJSON('/api/questions').then(function(resp){
      questions = resp || [];
    }).fail(function(xhr){
      alert('문제 불러오기 실패: ' + (xhr.responseJSON && xhr.responseJSON.detail || xhr.statusText));
    });
  }

  // 시작
  fetchQuestions().always(function(){
    if (!questions.length){
      $questionText.text('문제를 찾을 수 없습니다. 관리자에게 문의하세요.');
      $answerInput.prop('disabled', true);
      $submitBtn.prop('disabled', true);
      $hintBtn.prop('disabled', true);
      return;
    }
    loadQuestion(0);
    startTimer();
    $answerInput.focus();
  });
});
