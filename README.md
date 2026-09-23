# ML_Stat_Notes

Interactive, equation-first notes on statistical and computational methods for physicists.
Live at **https://dipakmunshi87.github.io/ML_Stat_Notes/**

Plain HTML, CSS and JavaScript: no build step, no framework, no network dependency. Equations are
pre-rendered text (KaTeX, fonts embedded in each topic's stylesheet) and every figure is *computed live*
in the browser rather than stored as an image.

## Layout

```
.
├── index.html          ← the master page: one card per topic
├── assets/site.css     ← style for the master page only
├── .nojekyll           ← tells GitHub Pages to serve every file untouched
│
├── mcmc/               ← topic 1: Monte Carlo samplers
│   ├── index.html
│   ├── metropolis.html  gibbs.html  hmc.html  nuts.html  nested.html
│   └── assets/         ← that topic's own css / js / media
│
└── time-series/        ← topic 2: time-series analysis
    ├── index.html
    ├── basics.html  dictionary.html  models.html  methods.html
    ├── examples.html  applications.html  errors.html  software.html
    └── assets/
```

Each topic folder is **self-contained**: its own `assets/`, its own stylesheet, its own JavaScript. Topics
never share files, so one can be changed, replaced or removed without touching any other. The only shared
thing is `index.html` at the root, which links to them.

Every link in the site is **relative**, so the whole tree works unchanged whether it is opened from disk,
served at the root of a domain, or served from a subdirectory as it is here.

## Adding a new topic

1. Put the new topic's folder at the top level, all lower-case with hyphens: `pinn/`,
   `variational-inference/`. It must contain its own `index.html` and its own `assets/`.
2. Open the root `index.html`, find the `<!-- TOPICS -->` block, copy one `<a class="tcard">` block and edit
   it: the `href` (the folder name with a trailing slash), the number in `<div class="mnum">`, the colour
   class (`c-blue`, `c-green`, `c-orange`, `c-purple`, `c-pink`), the title, the paragraph, and the chips in
   `<ul class="pages">`.
3. Inside the new topic, link back here with `href="../"` so readers can return to the master page.
4. Commit and push (below). The site updates about a minute later.

Nothing else needs changing — there is no index to regenerate and no configuration to keep in step.

## Publishing

One-time setup, from this folder:

```bash
git init
git add -A
git commit -m "Master page plus the MCMC and time-series topic sets"
git branch -M main
git remote add origin https://github.com/dipakmunshi87/ML_Stat_Notes.git
git push -u origin main
```

Then on GitHub: **Settings → Pages**. Source **Deploy from a branch**, branch `main`, folder `/ (root)`,
Save. The site appears at `https://dipakmunshi87.github.io/ML_Stat_Notes/` after a minute or two.

Every later change is just:

```bash
git add -A
git commit -m "what changed"
git push
```

`.nojekyll` is included so GitHub Pages serves every file untouched — without it, Pages runs the files
through Jekyll and ignores any directory beginning with an underscore.

## Checking before you push

Opening `index.html` from disk works exactly as the published site does — same relative links, same files.
So the useful check is simply to open it locally and click through every card and every top-bar link.

## Contents

| Topic | Pages |
|---|---|
| **Monte Carlo samplers** (`mcmc/`) | Metropolis–Hastings, Gibbs, Hamiltonian Monte Carlo, NUTS, nested sampling |
| **Time-series analysis** (`time-series/`) | Definitions, physics dictionary, models, methods, worked examples, further applications, error estimates, software |

## Credits

Prepared by Dipak Munshi with Claude (Anthropic), 2026.
Equation rendering by [KaTeX](https://katex.org) (MIT licence).
