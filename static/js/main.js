$(function(){
  function ts(s){ try { return new Date(s).toLocaleString('ko-KR'); } catch(e){ return s; } }

  $.getJSON('/api/ranking/list?limit=5')
    .done(function(resp){
      var $body = $('#lbTop5'), html = '';
      if (!resp.success || !resp.rankings || !resp.rankings.length){
        $body.html('<tr><td class="empty" colspan="4">아직 랭킹이 없습니다.</td></tr>');
        return;
      }
      for (var i=0; i<resp.rankings.length; i++){
        var row = resp.rankings[i];
        html += '<tr class="row-'+row.rank+'">' +
          '<td><strong>'+row.rank+'</strong></td>' +
          '<td>'+ $('<div>').text(row.username||'').html() +'</td>' +
          '<td>'+ (Number(row.score)||0).toLocaleString() +'</td>' +
          '<td>'+ ts(row.created_at) +'</td>' +
        '</tr>';
      }
      $body.html(html);
    })
    .fail(function(){
      $('#lbTop5').html('<tr><td class="empty" colspan="4">랭킹을 불러오지 못했습니다.</td></tr>');
    });

  $('.btn-start, .btn-rank').on('keypress', function(e){
    if (e.which === 13) this.click();
  });

  $('.nav-title').on('click', function(){
    location.href = "/";
  });
  
  $('.nav-mode').on('click', function(){
    location.href = "/";
  });
});