$(function(){
  var params = new URLSearchParams(location.search);
  if (params.get('just') === '1'){
    $('#lastSavedNotice').text('✅ 방금 기록이 저장되었습니다.').show();
  }

  function formatTs(s){
    try { return new Date(s).toLocaleString('ko-KR'); } catch(e){ return s; }
  }

  $.getJSON('/api/users?limit=100')           // ✅ 변경: /api/leaderboard -> /api/users
    .done(function(list){
      var $body = $('#leaderBody');
      if (!list || !list.length){
        $body.html('<tr><td colspan="4" class="empty">아직 등록된 기록이 없습니다.</td></tr>');
        return;
      }
      var html = '';
      for (var i=0;i<list.length;i++){
        var r = i+1;
        var row = list[i];
        html += '<tr>' +
          '<td><span class="badge-rank">'+r+'</span></td>' +
          '<td>'+ $('<div>').text(row.username || '').html() +'</td>' +         // ✅ username
          '<td>'+ (Number(row.total_score)||0).toLocaleString() +'</td>' +       // ✅ total_score
          '<td>'+ formatTs(row.created_at) +'</td>' +                             // ✅ created_at
        '</tr>';
      }
      $body.html(html);
    })
    .fail(function(xhr){
      $('#leaderBody').html('<tr><td colspan="4" class="empty">불러오기 실패: '+ (xhr.statusText || 'error') +'</td></tr>');
    });
});
