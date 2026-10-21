"""Assemble chapters/*.html into one note: substitute {{v:key}} values and {{fig:name|caption}} figures,
build TOC from <h2>/<h3>, write raw.html, then run mathrender.js -> quant_toolkit.html"""
import os, re, glob, json, base64, html as H, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
vals = json.load(open(os.path.join(HERE, "values.json")))
missing_v, missing_f = set(), set()

def sub_vals(s):
    def r(m):
        k = m.group(1)
        if k not in vals: missing_v.add(k); return f"<span style='background:#fdd'>??{k}??</span>"
        return str(vals[k])
    return re.sub(r"\{\{v:([^}]+)\}\}", r, s)

def sub_figs(s):
    def r(m):
        name, cap = m.group(1), m.group(2) or ""
        p = os.path.join(HERE, "figs", name + ".png")
        if not os.path.exists(p): missing_f.add(name); return f"<p style='background:#fdd'>MISSING FIG {name}</p>"
        b = base64.b64encode(open(p, "rb").read()).decode()
        return (f'<figure><img src="data:image/png;base64,{b}" alt="{H.escape(cap[:120])}">'
                f'<figcaption>{cap}</figcaption></figure>')
    return re.sub(r"\{\{fig:([\w\-]+)\|?(.*?)\}\}", r, s, flags=re.S)

chapters = sorted(glob.glob(os.path.join(HERE, "chapters", "*.html")))
body, toc = [], []
for i, path in enumerate(chapters):
    s = open(path, encoding="utf-8").read()
    s = sub_figs(sub_vals(s))
    sid = f"c{i}"
    m = re.search(r"<h2>(.*?)</h2>", s, re.S)
    title = re.sub(r"<[^>]+>", "", re.sub(r'<span class="n">(\d+)</span>', r"\1. ", m.group(1))).strip() if m else os.path.basename(path)
    part = re.search(r"<!--\s*PART:(.*?)-->", s)
    if part: toc.append(f'<div class="tocpart">{part.group(1).strip()}</div>')
    toc.append(f'<button class="navbtn" onclick="show({i})">{title}</button>')
    body.append((sid, s))
pages=[]
for i,(sid,s) in enumerate(body):
    prev = f'<button class="pg" onclick="show({i-1})">&larr; Previous</button>' if i>0 else '<span></span>'
    nxt = f'<button class="pg" onclick="show({i+1})">Next &rarr;</button>' if i<len(body)-1 else '<button class="pg" onclick="show(0)">&larr; Back to start</button>'
    pages.append(f'<section class="page" id="{sid}">\n{s}\n<div class="pager">{prev}<span class="pgn">page {i+1} of {len(body)}</span>{nxt}</div>\n</section>')
body=pages

tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
out = tpl.replace("<!--TOC-->", "\n".join(toc)).replace("<!--BODY-->", "\n".join(body))
raw = os.path.join(HERE, "raw.html"); open(raw, "w", encoding="utf-8").write(out)
print("chapters:", len(chapters), "| missing values:", sorted(missing_v), "| missing figs:", sorted(missing_f))
final = os.path.join(HERE, "quant_toolkit.html")
r = subprocess.run(["node", os.path.join(HERE, "mathrender.js"), raw, final], capture_output=True, text=True)
print(r.stdout, r.stderr[:3000])
txt = re.sub(r"<[^>]+>", " ", re.sub(r"<svg.*?</svg>|<style.*?</style>|<script.*?</script>", " ", open(final, encoding="utf-8").read(), flags=re.S))
print("words ~", len(txt.split()), "| size MB", round(os.path.getsize(final)/1e6, 2))
