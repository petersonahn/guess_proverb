$(function(){
  function ts(s){ try { return new Date(s).toLocaleString('ko-KR'); } catch(e){ return s; } }

  $.getJSON('/api/users/top?size=5')   // ← 여기!
    .done(function(list){
      var $body = $('#lbTop5'), html = '';
      if (!list || !list.length){
        $body.html('<tr><td class="empty" colspan="4">아직 랭킹이 없습니다.</td></tr>');
        return;
      }
      for (var i=0;i<list.length;i++){
        var r = i+1, row = list[i];
        html += '<tr class="row-'+r+'">' +
          '<td><strong>'+r+'</strong></td>' +
          '<td>'+ $('<div>').text(row.username||'').html() +'</td>' +
          '<td>'+ (Number(row.total_score)||0).toLocaleString() +'</td>' +
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

  $('.nav-mode').on('click', function(){
    location.href = "/index.html";
  });
});