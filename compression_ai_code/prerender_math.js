// Pre-render all TeX math in an HTML file to static SVG (no runtime JS needed).
// Usage: node prerender_math.js input.html output.html
const fs = require('fs');
const MJ = '/tmp/mjf/node_modules/mathjax-full/js';
const {mathjax} = require(MJ + '/mathjax.js');
const {TeX} = require(MJ + '/input/tex.js');
const {SVG} = require(MJ + '/output/svg.js');
const {liteAdaptor} = require(MJ + '/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require(MJ + '/handlers/html.js');
const {AllPackages} = require(MJ + '/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages});
const svg = new SVG({fontCache: 'local'});   // self-contained SVG per equation
const doc = mathjax.document('', {InputJax: tex, OutputJax: svg});

let n = 0;
function render(expr, display) {
  const node = doc.convert(expr.trim(), {display: display, em: 16, ex: 8, containerWidth: 800});
  n++;
  const cls = display ? 'mjx-eq mjx-display' : 'mjx-eq';
  return '<span class="' + cls + '">' + adaptor.outerHTML(node) + '</span>';
}

const [,, inPath, outPath] = process.argv;
let html = fs.readFileSync(inPath, 'utf8');

// order matters: display forms first, then inline
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, e) => render(e, true));
html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, e) => render(e, true));
html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, e) => render(e, false));
html = html.replace(/\$([^\$]+?)\$/g,       (_, e) => render(e, false));

// stylesheet MathJax needs for the SVG containers (once, in <head>)
const css = adaptor.textContent(svg.styleSheet(doc));
const style = '<style>' + css +
  '.mjx-eq{line-height:0}.mjx-display{display:block;text-align:center;margin:1em 0;overflow-x:auto;overflow-y:hidden}</style>';
html = html.replace('</head>', style + '\n</head>');

fs.writeFileSync(outPath, html);
console.error('rendered ' + n + ' equations -> ' + outPath);
