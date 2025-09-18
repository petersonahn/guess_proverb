// jQuery only logic
$(function() {
  // 문제 은행 (샘플 3개)
  var questions = [
    { text: '원숭이도 나무에서 ________', answers: ['떨어진다','떨어져'], hint: '아무리 능숙한 사람도 실수할 수 있다는 뜻.' },
    { text: '등잔 밑이 ________',           answers: ['어둡다','어두워'],   hint: '가까운 곳의 사정을 오히려 모른다는 뜻.' },
    { text: '가는 말이 고와야 오는 말이 ________', answers: ['곱다'],      hint: '상대에게 좋게 해야 그도 좋게 한다는 뜻.' }
  ];

  // 요소 캐시
  var $answerInput = $('.answer-input');
  var $hintBtn = $('.btn-hint');
  var $submitBtn = $('.btn-submit');
  var $reloadBtn = $('.btn-reload');
  var $currentScore = $('#currentScore');
  var $timeFill = $('#timeFill');
  var $timeDisplay = $('#timeDisplay');
  var $correctCount = $('#correctCount');
  var $streakCount = $('#streakCount');
  var $questionText = $('#questionText');

  // 상태
  var currentScore = 0;
  var scorePerCorrect = 100;
  var correctCount = 0;
  var streakCount = 0;
  var currentIndex = 0;

  var totalTime = 10;   // 데모용 10초
  var remainingTime = totalTime;
  var timerInterval = null;

  // 문제 로드/전환
  function loadQuestion(idx) {
    currentIndex = idx;
    $questionText.text(questions[currentIndex].text);
    $reloadBtn.hide();
    $submitBtn.show();
    $hintBtn.show();
    $answerInput.show();
    $answerInput.val('');
    $answerInput.focus();
  }
  function nextQuestion() {
    var next = (currentIndex + 1) % questions.length;
    loadQuestion(next);
  }

  // 정답 판정
  function normalize(s) {
    return $.trim(s).replace(/\s+/g,'').replace(/[.,!?~]/g,'');
  }
  function isCorrectAnswer(input) {
    var norm = normalize(input);
    for (var i=0;i<questions[currentIndex].answers.length;i++){
      if (normalize(questions[currentIndex].answers[i]) === norm) return true;
    }
    return false;
  }

  // 타이머
  function startTimer() {
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

  // 게임오버
  function gameOver() {
    $timeDisplay.text("0:00");
    showCustomAlert('⏰ 시간 종료!', '게임이 끝났습니다!\n총 점수: ' + currentScore.toLocaleString() + '점\n정답 개수: ' + correctCount + '개', 'info');
    showRankModal(currentScore);
    $answerInput.hide();
    $submitBtn.hide();
    $hintBtn.hide();
    $reloadBtn.show();
  }

  // 점수/통계
  function updateScore(points) {
    currentScore += points;
    correctCount += 1;
    streakCount += 1;

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
    var hint = questions[currentIndex].hint || '이 문제에는 힌트가 없습니다.';
    showCustomAlert('💡 힌트', hint, 'info');
  });

  // 제출
  $submitBtn.on('click', function(){
    var answer = $answerInput.val();
    if ($.trim(answer) === '') {
      showCustomAlert('⚠️ 알림','답을 입력해주세요.','warning');
      $answerInput.focus();
      return;
    }
    if (isCorrectAnswer(answer)) {
      updateScore(scorePerCorrect);
      showCustomAlert('🎉 정답!','축하합니다! (+' + scorePerCorrect + '점)\n연속 ' + streakCount + '회 정답!','success');
      createSuccessParticles();
      setTimeout(function(){ nextQuestion(); }, 700);
    } else {
      resetStreak();
      showCustomAlert('❌ 틀렸습니다','다시 시도해보세요!','error');
      $answerInput.select();
    }
  });

  // 다시하기
  $reloadBtn.on('click', function(){
    location.reload();
  });

  // Enter 제출
  $answerInput.on('keypress', function(e){
    if (e.which === 13) { $submitBtn.click(); }
  });

  // 알림/이펙트
  function showCustomAlert(title, message, type) {
    var emoji = type === 'success' ? '🎉' : (type === 'error' ? '❌' : (type === 'warning' ? '⚠️' : '💡'));
    alert(emoji + ' ' + title + '\n\n' + message);
  }
  function createSuccessParticles(){ console.log('🎉 Success particles!'); }

  // ===== 랭킹 저장/로드 (localStorage) =====
  var LS_KEY = 'proverb_master_leaderboard';
  function getLeaderboard(){
    try {
      var raw = localStorage.getItem(LS_KEY);
      if (!raw) return [];
      var arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return [];
      return arr;
    } catch(e){ return []; }
  }
  function saveLeaderboard(arr){
    localStorage.setItem(LS_KEY, JSON.stringify(arr));
  }
  function addLeaderboardEntry(name, score){
    var list = getLeaderboard();
    list.push({ name: String(name).slice(0,20), score: Number(score)||0, ts: Date.now() });
    // 점수 내림차순, 동점자는 먼저 기록한 사람이 위
    list.sort(function(a,b){ return b.score - a.score || a.ts - b.ts; });
    // 상위 100개만 유지
    if (list.length > 100) list = list.slice(0,100);
    saveLeaderboard(list);
  }

  // === ID 중복 방지 ===
  function isNameTaken(name){
    var lower = String(name).toLowerCase();
    var list = getLeaderboard();
    for (var i=0;i<list.length;i++){
      if (String(list[i].name||'').toLowerCase() === lower) return true;
    }
    return false;
  }
  function findUniqueName(base){
    var candidate = base;
    var i = 2;
    while (isNameTaken(candidate) && i < 10000){
      candidate = base + '-' + i;
      i++;
    }
    return candidate;
  }

  // ===== 랭킹 모달 =====
  var $rankModal = $('#rankModal');
  var $rankClose = $('#rankCloseBtn');
  var $rankLater = $('#rankLaterBtn');
  var $rankSave = $('#rankSaveBtn');
  var $playerName = $('#playerName');
  var $finalScoreText = $('#finalScoreText');

  function showRankModal(finalScore){
    $finalScoreText.text(finalScore.toLocaleString());
    $playerName.val('');
    // flex로 강제 후 fadeIn => 중앙 정렬 유지
    $rankModal.css('display','flex').hide().fadeIn(160);
    setTimeout(function(){ $playerName.focus(); }, 50);
  }
  function hideRankModal(){
    $rankModal.fadeOut(120);
  }
  $rankClose.on('click', hideRankModal);
  $rankLater.on('click', function(){
    hideRankModal();
  });
  $rankSave.on('click', function(){
    var name = $.trim($playerName.val());
    if (!name){
      alert('이름(또는 ID)을 입력해주세요.');
      $playerName.focus();
      return;
    }

    // 중복 방지 로직
    var unique = findUniqueName(name);
    if (unique !== name){
      alert('이미 존재하는 ID입니다.\n다시 입력해주세요.');
      return false;
    }

    addLeaderboardEntry(unique, currentScore);
    // 저장 후 랭킹 페이지로 이동
    window.location.href = './rankings.html?just=1';
  });

  // 시작
  loadQuestion(0);
  startTimer();
  $answerInput.focus();
});
