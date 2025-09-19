$(function(){
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
  function formatTs(ts){
    try {
      return new Date(ts).toLocaleString('ko-KR');
    } catch(e){
      return '';
    }
  }
  // 쿼리 파라미터 just=1 이면 방금 저장 알림
  var params = new URLSearchParams(location.search);
  if (params.get('just') === '1'){
    $('#lastSavedNotice').text('✅ 방금 기록이 저장되었습니다.').show();
  }

  var list = getLeaderboard();
  var $body = $('#leaderBody');
  if (!list.length){
    $body.html('<tr><td colspan="4" class="empty">아직 등록된 기록이 없습니다.</td></tr>');
    return;
  }
  var html = '';
  for (var i=0;i<list.length;i++){
    var r = i+1;
    var row = list[i];
    html += '<tr>' +
      '<td><span class="badge-rank">'+r+'</span></td>' +
      '<td>'+ $('<div>').text(row.name || '').html() +'</td>' +
      '<td>'+ (Number(row.score)||0).toLocaleString() +'</td>' +
      '<td>'+ formatTs(row.ts) +'</td>' +
    '</tr>';
  }
  $body.html(html);
});