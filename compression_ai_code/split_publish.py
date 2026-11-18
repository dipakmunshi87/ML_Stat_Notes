"""Split the full compression note into two paged decks (Part A = main note,
Part B = appendices), linked from a small front/index page carrying OpenGraph
tags + a LinkedIn preview card. Cross-file deep links via location.hash."""
import os, re, base64

SRC  = "/home/claude/compress/compression_ai_note.html"
DIR  = "/home/claude/compress"
PA   = "compression_ai_part_A.html"
PB   = "compression_ai_part_B.html"
IDX  = "compression_ai.html"
REPO = "https://dipakmunshi87.github.io/ML_Stat_Notes"

DECK_CSS = r"""
<style>
html,body{height:100%}
.topnav{position:sticky;top:0;z-index:46;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line,#e3e8ee);padding:9px 0;margin:0 0 4px;display:flex;align-items:center;}
.topnav .inner{max-width:820px;margin:0 auto;padding:0 22px;display:flex;gap:10px;align-items:center;width:100%;}
.topnav .brand{color:var(--acc,#3D7EBF);font-weight:800;letter-spacing:.02em;font-size:.92rem;}
.topnav .home{color:var(--mut,#5b6b7a);text-decoration:none;font-size:.84rem;border:1px solid var(--line,#e3e8ee);
  border-radius:999px;padding:5px 12px;font-weight:650;white-space:nowrap;}
.topnav .home:hover{border-color:var(--acc,#3D7EBF);color:var(--acc,#3D7EBF);}
.otherpart{color:var(--acc2,#245a95);text-decoration:none;font-size:.84rem;border:1px solid var(--line,#e3e8ee);
  border-radius:999px;padding:5px 12px;font-weight:650;white-space:nowrap;}
.otherpart:hover{border-color:var(--acc,#3D7EBF);}
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
.pager a.endlink{border:1px solid var(--acc,#3D7EBF);background:var(--acc,#3D7EBF);color:#fff;border-radius:10px;
  padding:9px 18px;font-size:.92rem;font-weight:650;text-decoration:none;}
.xref{color:var(--acc,#3D7EBF);cursor:pointer;}
a.xfile{color:var(--acc,#3D7EBF);text-decoration:none;border-bottom:1px dotted var(--acc,#3D7EBF);}
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
  window.scrollTo({top:0,behavior:'smooth'});
}
function toggleTOC(){var t=document.getElementById('toc'),b=document.getElementById('tocback');var o=t.classList.toggle('open');b.classList.toggle('open',o);}
function closeTOC(){var t=document.getElementById('toc'),b=document.getElementById('tocback');if(t)t.classList.remove('open');if(b)b.classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(cur-1);if(e.key==='ArrowRight')show(cur+1);if(e.key==='Escape')closeTOC();});
function _hashpage(){var m=(location.hash||'').match(/p(\\d+)/);return m?parseInt(m[1]):0;}
window.addEventListener('hashchange',function(){show(_hashpage());});
show(_hashpage());
</script>
"""

def pager(endlink=None):
    nxt = ('<button onclick="show(cur+1)" id="next">Next &rarr;</button>' if endlink is None
           else '<a class="endlink" href="%s">%s</a>' % endlink)
    return ('<div class="pager"><button onclick="show(cur-1)" id="prev">&larr; Prev</button>' + nxt + '</div>')

s = open(SRC, encoding="utf-8").read()
head   = s[:s.find("</head>")]
header = s[s.find("<header>"):s.find("</header>")+len("</header>")]
wrap   = s.split('<div class="wrap">',1)[1]
wrap   = wrap[:wrap.rfind('</div>')]

parts = re.split(r'(?=<h2)', wrap)
front, secs = parts[0], parts[1:]
# index secs: 0..22  (0=§1 ... 9=§10, 10=Conclusion, 11=§11 Term, 12=ApxA ... 21=ApxJ, 22=References)

def sec_title(sec):
    m = re.match(r'<h2[^>]*>(.*?)</h2>', sec, re.S)
    t = re.sub(r'<[^>]+>', '', m.group(1)).replace('&nbsp;',' ').strip()
    return re.sub(r'\s+',' ', t)

REFSEC = secs[-1]               # References (shared, duplicated into both parts)
A_secs = secs[0:12]             # §1..§10, Conclusion, §11 Terminology
B_secs = secs[12:22]            # Appendix A..J

# ---- page maps ----
# Part A: p0 front, p1..p12 = A_secs, p13 = References
A_refpage = 1 + len(A_secs)                       # 13
# Part B: p0 title, p1..p10 = B_secs, p11 = References
B_refpage = 1 + len(B_secs)                       # 11

sec_to_Apage = {}                                 # §N -> Part A page
for i, sec in enumerate(A_secs, 1):
    m = re.match(r'<h2[^>]*><span class="n">(\d+)</span>', sec)
    if m: sec_to_Apage[int(m.group(1))] = i
appx_to_Bpage = {}                                # 'A'.. -> Part B page
for i, sec in enumerate(B_secs, 1):
    m = re.search(r'Appendix ([A-J])', sec)
    if m: appx_to_Bpage[m.group(1)] = i

def rewrite(txt, part, refpage):
    # citations -> local References page
    txt = re.sub(r'href="#r\d+"', 'href="javascript:void(0)" onclick="show(%d)"' % refpage, txt)
    # protect the h2 heading from Appendix-link rewriting
    mo = re.match(r'(<h2.*?</h2>)', txt, re.S)
    head_h2, body = (mo.group(1), txt[mo.end():]) if mo else ("", txt)
    # section cross-refs  &sect;N
    def sref(m):
        n = int(m.group(2))
        if part == "A" and n in sec_to_Apage:
            return '<a class="xref" onclick="show(%d)">%s</a>' % (sec_to_Apage[n], m.group(0))
        if part == "B" and n in sec_to_Apage:
            return '<a class="xfile" href="%s#p%d">%s</a>' % (PA, sec_to_Apage[n], m.group(0))
        return m.group(0)
    body = re.sub(r'(&sect;)(\d+)', sref, body)
    # appendix cross-refs  "Appendix X"
    def aref(m):
        x = m.group(1)
        if x not in appx_to_Bpage: return m.group(0)
        if part == "B":
            return '<a class="xref" onclick="show(%d)">Appendix %s</a>' % (appx_to_Bpage[x], x)
        return '<a class="xfile" href="%s#p%d">Appendix %s</a>' % (PB, appx_to_Bpage[x], x)
    body = re.sub(r'Appendix ([A-J])\b', aref, body)
    return head_h2 + body

def topnav(brand, other_label, other_file):
    return ('<div class="topnav"><div class="inner"><span class="brand">%s</span>'
            '<a class="home" href="%s">&#8962; Front</a>'
            '<a class="otherpart" href="%s">%s &rarr;</a>'
            '<button class="tocbtn" onclick="toggleTOC()">&#9776; Contents</button>'
            '<span class="pagenum" id="pagenum"></span></div></div>'
            % (brand, IDX, other_file, other_label))

def toc(titles):
    t = '<div class="tocback" id="tocback" onclick="toggleTOC()"></div><div class="toc" id="toc">'
    for i, ti in enumerate(titles):
        t += '<button class="navbtn" onclick="show(%d)">%s</button>' % (i, ti)
    return t + '</div>'

def build_deck(outfile, brand, pages, titles, other_label, other_file):
    N = len(pages)
    html = (head + DECK_CSS + "</head>\n<body>\n"
            + topnav(brand, other_label, other_file) + toc(titles)
            + '<div class="wrap">\n' + "\n".join(pages) + '\n</div>'
            + (SCRIPT_TMPL % N) + "\n</body></html>")
    open(os.path.join(DIR, outfile), "w", encoding="utf-8").write(html)
    print("wrote", outfile, f"({len(html)/1024/1024:.2f} MB)  {N} pages")

# ---------- PART A ----------
A_pages = ['<section class="page on" id="p0">' + header + rewrite(front, "A", A_refpage) + pager() + '</section>']
A_titles = ["0. Overview"]
for i, sec in enumerate(A_secs, 1):
    A_pages.append('<section class="page" id="p%d">' % i + rewrite(sec, "A", A_refpage) + pager() + '</section>')
    A_titles.append("%d. %s" % (i, sec_title(sec)))
# References page: link onward to Part B
A_pages.append('<section class="page" id="p%d">' % A_refpage + rewrite(REFSEC, "A", A_refpage)
               + pager(endlink=(PB, "Continue to Part B: Appendices &rarr;")) + '</section>')
A_titles.append("%d. References" % A_refpage)
build_deck(PA, "Data Compression &amp; AI &middot; Part A", A_pages, A_titles,
           "Part B: Appendices", PB)

# ---------- PART B ----------
B_header = ('<header><div class="eyebrow">A reproducible technical note &middot; Part B</div>'
            '<h1>Data compression and AI &mdash; Appendices</h1>'
            '<p class="sub">Worked 1D and 2D pen-and-paper examples behind the main note (Part A): '
            'the score, autoencoders, reconstruction vs realization, VAEs, stochastic variational '
            'inference, latent variables and amortization, the ELBO, normalizing flows, related '
            'fields, and ICA.</p>'
            '<p class="sub"><a class="xfile" href="%s">&larr; Back to Part A (the main note)</a> '
            '&middot; <a class="xfile" href="%s">front page</a></p></header>' % (PA, IDX))
B_pages = ['<section class="page on" id="p0">' + B_header + pager() + '</section>']
B_titles = ["0. Overview"]
for i, sec in enumerate(B_secs, 1):
    B_pages.append('<section class="page" id="p%d">' % i + rewrite(sec, "B", B_refpage) + pager() + '</section>')
    B_titles.append("%d. %s" % (i, sec_title(sec)))
B_pages.append('<section class="page" id="p%d">' % B_refpage + rewrite(REFSEC, "B", B_refpage)
               + pager(endlink=(PA, "&larr; Back to Part A")) + '</section>')
B_titles.append("%d. References" % B_refpage)
build_deck(PB, "Data Compression &amp; AI &middot; Part B", B_pages, B_titles,
           "Part A: Main note", PA)

# ---------- FRONT / INDEX ----------
prev_b64 = base64.b64encode(open(os.path.join(DIR,"compression_ai_preview.png"),"rb").read()).decode()
idx = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Compression and AI &mdash; a reproducible technical note</title>
<meta name="description" content="Why learning is compression: from Shannon and MDL through PCA, MOPED, autoencoders, VAEs and normalizing flows, with weak-lensing examples and pen-and-paper demos. By Dr. Dipak Munshi.">
<meta property="og:type" content="article">
<meta property="og:title" content="Data Compression and AI">
<meta property="og:description" content="Why learning is compression — from Shannon and MDL through PCA, MOPED, autoencoders, VAEs and normalizing flows, with weak-lensing examples and pen-and-paper demos.">
<meta property="og:url" content="REPO/compression_ai.html">
<meta property="og:image" content="REPO/compression_ai_preview.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Data Compression and AI">
<meta name="twitter:description" content="Why learning is compression — from Shannon and MDL through PCA, MOPED, autoencoders, VAEs and normalizing flows, with weak-lensing examples.">
<meta name="twitter:image" content="REPO/compression_ai_preview.png">
<style>
:root{--acc:#3D7EBF;--acc2:#245a95;--ink:#1c2530;--mut:#5b6b7a;--line:#e3e8ee;--bg:#f6f8fb;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6;}
.wrap{max-width:820px;margin:0 auto;padding:30px 22px 60px;}
.card{background:#fff;border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 8px 30px rgba(20,50,90,.07);}
.card img{display:block;width:100%;height:auto;border-bottom:1px solid var(--line);}
.body{padding:26px 30px 32px;}
.eyebrow{color:var(--mut);font-size:.9rem;letter-spacing:.02em;}
h1{font-size:1.9rem;margin:.25em 0 .35em;color:var(--acc2);line-height:1.2;}
p.lead{font-size:1.06rem;color:var(--ink);margin:.4em 0 1.1em;}
.byline{color:var(--mut);font-size:.95rem;margin-bottom:1.4em;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0 4px;}
@media(max-width:560px){.grid{grid-template-columns:1fr;}}
a.part{display:block;text-decoration:none;border:1px solid var(--line);border-radius:13px;padding:18px 20px;
 transition:.15s;background:#fff;}
a.part:hover{border-color:var(--acc);box-shadow:0 6px 20px rgba(61,126,191,.12);transform:translateY(-1px);}
a.part .t{color:var(--acc2);font-weight:750;font-size:1.08rem;}
a.part .d{color:var(--mut);font-size:.9rem;margin-top:5px;}
a.part .pill{display:inline-block;background:#eef4fb;color:var(--acc2);font-size:.74rem;font-weight:700;
 border-radius:999px;padding:2px 9px;margin-bottom:8px;letter-spacing:.03em;}
.foot{color:var(--mut);font-size:.84rem;margin-top:22px;border-top:1px solid var(--line);padding-top:16px;}
.foot a{color:var(--acc2);}
</style></head>
<body><div class="wrap"><div class="card">
<img src="compression_ai_preview.png" alt="Data Compression and AI">
<div class="body">
<div class="eyebrow">A reproducible technical note &middot; ML &amp; Stat Notes series</div>
<h1>Data compression and AI: two names for one thing</h1>
<p class="lead">Why learning <em>is</em> compression &mdash; from Shannon and MDL through PCA, MOPED,
autoencoders, VAEs and normalizing flows &mdash; with weak-lensing examples, a worked demo, and
1D/2D pen-and-paper exercises throughout. Every figure is generated by the code described in the note.</p>
<div class="byline">Dr. Dipak Munshi &nbsp;&middot;&nbsp; Advanced Research Fellow, Imperial College London</div>
<div class="grid">
<a class="part" href="PA_FILE"><span class="pill">PART A</span>
<div class="t">The main note &rarr;</div>
<div class="d">&sect;1&ndash;10 + conclusion + terminology: learning as compression, PCA / MOPED,
neural compression, the demo, the information view, generative models as compressors, honest caveats.</div></a>
<a class="part" href="PB_FILE"><span class="pill">PART B</span>
<div class="t">The appendices &rarr;</div>
<div class="d">Appendices A&ndash;J: the score, autoencoders by hand, reconstruction vs realization,
VAEs, stochastic variational inference, latent variables &amp; amortization, the ELBO, normalizing
flows, related fields, and ICA.</div></a>
</div>
<div class="foot">Read in order &mdash; Part A links through to Part B at the end &mdash; or jump straight
to either. Best viewed on a wide screen. &nbsp;&middot;&nbsp; <a href="PA_FILE">Start reading &rarr;</a></div>
</div></div></div></body></html>"""
idx = idx.replace("REPO", REPO).replace("PA_FILE", PA).replace("PB_FILE", PB)
open(os.path.join(DIR, IDX), "w", encoding="utf-8").write(idx)
print("wrote", IDX, f"({len(idx)/1024:.0f} KB)")
