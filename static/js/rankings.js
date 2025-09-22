$(function(){
  var params = new URLSearchParams(location.search);
  if (params.get('just') === '1'){
    $('#lastSavedNotice').text('✅ 방금 기록이 저장되었습니다.').show();
  }

  function formatTs(s){
    try { return new Date(s).toLocaleString('ko-KR'); } catch(e){ return s; }
  }

  // FastAPI 랭킹 API 호출
  $.getJSON('/api/ranking/list?limit=100')
    .done(function(resp){
      var $body = $('#leaderBody');
      if (!resp.success || !resp.rankings || !resp.rankings.length){
        $body.html('<tr><td colspan="4" class="empty">아직 등록된 기록이 없습니다.</td></tr>');
        return;
      }
      
      var html = '';
      for (var i = 0; i < resp.rankings.length; i++){
        var row = resp.rankings[i];
        var rankBadge = '';
        
        // 순위별 배지 스타일
        if (row.rank === 1) {
          rankBadge = '<span class="badge-rank gold">🥇 ' + row.rank + '</span>';
        } else if (row.rank === 2) {
          rankBadge = '<span class="badge-rank silver">🥈 ' + row.rank + '</span>';
        } else if (row.rank === 3) {
          rankBadge = '<span class="badge-rank bronze">🥉 ' + row.rank + '</span>';
        } else {
          rankBadge = '<span class="badge-rank">' + row.rank + '</span>';
        }
        
        html += '<tr>' +
          '<td>' + rankBadge + '</td>' +
          '<td>' + $('<div>').text(row.username || '').html() + '</td>' +
          '<td>' + (Number(row.score) || 0).toLocaleString() + '점</td>' +
          '<td>' + formatTs(row.created_at) + '</td>' +
        '</tr>';
      }
      $body.html(html);
    })
    .fail(function(xhr){
      console.error('랭킹 로딩 실패:', xhr);
      $('#leaderBody').html('<tr><td colspan="4" class="empty">불러오기 실패: '+ (xhr.responseJSON?.detail || xhr.statusText || 'error') +'</td></tr>');
    });

  // 메인 페이지로 이동
  $('.nav-title').on('click', function() {
    location.href = '/';
  });
});
