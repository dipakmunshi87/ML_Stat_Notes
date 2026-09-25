
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide')), i=0;
  function show(n){
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach(function(s,k){s.classList.toggle('active',k===i)});
    document.getElementById('count').textContent=(i+1)+' / '+slides.length;
    document.getElementById('progress').style.width=(100*(i+1)/slides.length)+'%';
    try{history.replaceState(null,'','#s'+(i+1))}catch(e){}
    document.querySelectorAll('video').forEach(function(v){ if(!slides[i].contains(v)) v.pause(); });
    var v=slides[i].querySelector('video'); if(v){ var p=v.play(); if(p&&p.catch)p.catch(function(){}); }
    window.dispatchEvent(new CustomEvent('slidechange',{detail:i}));
  }
  function fromHash(){var m=/^#s?(\d+)$/.exec(location.hash||'');return m?parseInt(m[1],10)-1:null}
  var h=fromHash(); show(h===null?0:h);
  window.addEventListener('hashchange',function(){var k=fromHash();if(k!==null&&k!==i)show(k)});
  document.getElementById('prev').onclick=function(){show(i-1)};
  document.getElementById('home').onclick=function(e){e.preventDefault();show(0)};
  var fs=19;function setFs(v){fs=Math.max(14,Math.min(30,v));document.documentElement.style.fontSize=fs+'px';window.dispatchEvent(new CustomEvent('slidechange',{detail:i}))}
  document.getElementById('zoomIn').onclick=function(){setFs(fs+2)};
  document.getElementById('zoomOut').onclick=function(){setFs(fs-2)};
  document.getElementById('next').onclick=function(){show(i+1)};
  document.getElementById('notesBtn').onclick=function(){document.body.classList.toggle('show-notes')};
  document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT')return;
    if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){e.preventDefault();show(i+1)}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(i-1)}
    else if(e.key==='Home')show(0); else if(e.key==='End')show(slides.length-1);
    else if(e.key==='n'||e.key==='N')document.body.classList.toggle('show-notes');
  });
  var x0=null,y0=null;
  document.addEventListener('touchstart',function(e){if(e.target.tagName==='CANVAS'||e.target.tagName==='INPUT'){x0=null;return}x0=e.touches[0].clientX;y0=e.touches[0].clientY},{passive:true});
  document.addEventListener('touchend',function(e){if(x0===null)return;var dx=e.changedTouches[0].clientX-x0,dy=e.changedTouches[0].clientY-y0;
    if(Math.abs(dx)>60&&Math.abs(dx)>1.5*Math.abs(dy))show(i+(dx<0?1:-1));x0=null},{passive:true});
  window.__deck={show:show,get i(){return i}};
})();

