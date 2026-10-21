// Render $...$ and $$...$$ TeX in an HTML file to MathJax SVG (same markup as the existing notes).
const fs = require('fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages, inlineMath: [['$', '$']], displayMath: [['$$', '$$']]});
const svg = new SVG({fontCache: 'local'});
const html = mathjax.document('', {InputJax: tex, OutputJax: svg});

const [,, inFile, outFile] = process.argv;
let src = fs.readFileSync(inFile, 'utf8');

// protect <pre> and <code> blocks
const holds = [];
src = src.replace(/<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>/g, m => { holds.push(m); return `\u0000H${holds.length-1}\u0000`; });

let nDisp = 0, nInl = 0, errs = 0;
function render(texStr, display) {
  try {
    const node = html.convert(texStr, {display});
    let out = adaptor.outerHTML(node);
    if (display) { nDisp++; return `<span class="mjx-eq mjx-display">${out}</span>`; }
    nInl++; return `<span class="mjx-eq">${out}</span>`;
  } catch (e) { errs++; console.error('MATH ERROR:', texStr.slice(0,80), e.message); return `<code>${texStr}</code>`; }
}
// display first
src = src.replace(/\$\$([\s\S]+?)\$\$/g, (m, t) => render(t.trim(), true));
// inline: $...$ not spanning newlines excessively; allow escaped \$ inside text as literal
src = src.replace(/\\\$/g, '\u0001DOLLAR\u0001');
src = src.replace(/\$([^$\n]+?)\$/g, (m, t) => render(t.trim(), false));
src = src.replace(/\u0001DOLLAR\u0001/g, '$');
src = src.replace(/\u0000H(\d+)\u0000/g, (m, i) => holds[+i]);

// the shared SVG styles
const css = adaptor.textContent(svg.styleSheet(html));
src = src.replace('/*MJXCSS*/', css);
fs.writeFileSync(outFile, src);
console.log(`rendered ${nDisp} display + ${nInl} inline equations, ${errs} errors`);
if (/\$[^$]{1,60}\$/.test(src.replace(/<svg[\s\S]*?<\/svg>/g,''))) console.error('WARNING: possible unrendered $ left');
