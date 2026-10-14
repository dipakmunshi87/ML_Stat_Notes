"""Paged (slide-deck) version of the U-Net note: same button navigation as the
other LinkedIn presentations. Adapted to this note's <header>+<div class='wrap'>
structure, with citation [#rN] links routed to the References slide."""
import os, re
SRC="/home/claude/compress/compression_ai_note.html"
OUT="/home/claude/compress/compression_ai_deck.html"
BRAND="Data Compression & AI"

DECK_CSS = r"""
<style>
html,body{height:100%}
.topnav{position:sticky;top:0;z-index:46;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line,#e3e8ee);padding:9px 0;margin:0 0 4px;display:flex;align-items:center;}
.topnav .inner{max-width:820px;margin:0 auto;padding:0 22px;display:flex;gap:10px;align-items:center;width:100%;}
.topnav .brand{color:var(--acc,#3D7EBF);font-weight:800;letter-spacing:.02em;font-size:.92rem;}
.tocbtn{border:1px solid var(--line,#e3e8ee);background:#fff;color:#245a95;border-radius:999px;
  padding:5px 14px;font-size:.84rem;cursor:pointer;font-weight:650;transition:.15s;white-space:nowrap;}
.tocbtn:hover{border-color:var(--acc,#3D7EBF);color:var(--acc,#3D7EBF);}
.pagenum{margin-left:auto;color:var(--mut,#5b6b7a);font-size:.84rem;font-variant-numeric:tabular-nums;}
.tocback{position:fixed;inset:0;background:rgba(20,40,70,.28);z-index:45;display:none;}
.tocback.open{display:block;}
.toc{display:none;position:fixed;top:52px;left:50%;transform:translateX(-50%);
  width:min(820px,93vw);max-height:74vh;overflow-y:auto;background:#fff;
  border:1px solid var(--line,#e3e8ee);border-radius:14px;box-shadow:0 12px 40px rgba(20,50,90,.22);
  padding:12px;z-index:47;gap:6px;}
.toc.open{display:flex;flex-wrap:wrap;}
.navbtn{border:1px solid var(--line,#e3e8ee);background:#fff;color:var(--mut,#5b6b7a);border-radius:999px;
  padding:5px 12px;font-size:.82rem;cursor:pointer;transition:.15s;text-align:left;}
.navbtn:hover{border-color:var(--acc,#3D7EBF);color:#245a95;}
.navbtn.on{background:var(--acc,#3D7EBF);color:#fff;border-color:var(--acc,#3D7EBF);font-weight:650;}
.page{display:none;animation:fade .25s ease;}
.page.on{display:block;}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.pager{display:flex;justify-content:space-between;margin-top:30px;gap:10px;border-top:1px solid var(--line,#e3e8ee);padding-top:18px;}
.pager button{border:1px solid var(--line,#e3e8ee);background:#fff;color:#245a95;border-radius:10px;
  padding:9px 18px;font-size:.92rem;cursor:pointer;font-weight:600;}
.pager button:hover{border-color:var(--acc,#3D7EBF);}
.pager button:disabled{opacity:.35;cursor:default;}
.xref{color:var(--acc,#3D7EBF);cursor:pointer;}
</style>
"""

SCRIPT_TMPL = """
<script>
var cur=0, N=%d;
function show(k){
  if(k<0)k=0; if(k>=N)k=N-1; cur=k;
  document.querySelectorAll('.page').forEach((p,i)=>p.classList.toggle('on',i===k));
  document.querySelectorAll('.toc .navbtn').forEach((b,i)=>b.classList.toggle('on',i===k));
  var pn=document.getElementById('pagenum'); if(pn) pn.textContent=(k+1)+' / '+N;
  document.querySelectorAll('#prev').forEach(b=>b.disabled=(k===0));
  document.querySelectorAll('#next').forEach(b=>b.disabled=(k===N-1));
  closeTOC();
  if(window.MathJax&&MathJax.typesetPromise){MathJax.typesetPromise();}
  window.scrollTo({top:0,behavior:'smooth'});
}
function toggleTOC(){var t=document.getElementById('toc'),b=document.getElementById('tocback');var o=t.classList.toggle('open');b.classList.toggle('open',o);}
function closeTOC(){var t=document.getElementById('toc'),b=document.getElementById('tocback');if(t)t.classList.remove('open');if(b)b.classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(cur-1);if(e.key==='ArrowRight')show(cur+1);if(e.key==='Escape')closeTOC();});
show(0);
</script>
"""

def pager():
    return ('<div class="pager"><button onclick="show(cur-1)" id="prev">&larr; Prev</button>'
            '<button onclick="show(cur+1)" id="next">Next &rarr;</button></div>')

s = open(SRC, encoding="utf-8").read()
head = s[:s.find("</head>")]
header = s[s.find("<header>"):s.find("</header>")+len("</header>")]
wrap = s.split('<div class="wrap">',1)[1]
wrap = wrap[:wrap.rfind('</div>')]      # content inside .wrap

parts = re.split(r'(?=<h2)', wrap)
front, secs = parts[0], parts[1:]
N = 1 + len(secs)
refpage = N-1  # References is the last section

titles = ["Title &amp; overview"]
for sec in secs:
    m = re.match(r'<h2[^>]*>(.*?)</h2>', sec, re.S)
    t = re.sub(r'<[^>]+>', '', m.group(1)).replace('&nbsp;',' ').strip()
    t = re.sub(r'&mdash;.*$','',t).strip()   # trim trailing dash clauses
    titles.append(re.sub(r'\s+',' ',t) or "References")

def rewrite(txt):
    # citation links #rN -> References slide
    txt = re.sub(r'href="#r\d+"', 'href="javascript:void(0)" onclick="show(%d)"' % refpage, txt)
    # section cross-refs  §N / &sect;N -> that slide
    txt = re.sub(r'(&sect;)(\d+)',
                 lambda m: '<a class="xref" onclick="show(%d)">%s</a>' % (int(m.group(2)), m.group(0))
                           if int(m.group(2)) < N else m.group(0), txt)
    return txt

pages = ['<section class="page on" id="p0">' + header + rewrite(front) + pager() + '</section>']
for i, sec in enumerate(secs, 1):
    pages.append('<section class="page" id="p%d">' % i + rewrite(sec) + pager() + '</section>')

toc = '<div class="tocback" id="tocback" onclick="toggleTOC()"></div><div class="toc" id="toc">'
for i,t in enumerate(titles):
    toc += '<button class="navbtn" onclick="show(%d)">%s. %s</button>' % (i, i, t)
toc += '</div>'

topnav = ('<div class="topnav"><div class="inner"><span class="brand">%s</span>'
          '<button class="tocbtn" onclick="toggleTOC()">&#9776; Contents</button>'
          '<span class="pagenum" id="pagenum"></span></div></div>' % BRAND)

html = (head + DECK_CSS + "</head>\n<body>\n" + topnav + toc
        + '<div class="wrap">\n' + "\n".join(pages) + '\n</div>'
        + (SCRIPT_TMPL % N) + "\n</body></html>")
open(OUT,"w",encoding="utf-8").write(html)
print("wrote", os.path.basename(OUT), f"({len(html)/1024:.0f} KB)  {N} slides")
