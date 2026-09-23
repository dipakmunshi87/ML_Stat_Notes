# MCMC demo set — Monte Carlo samplers, from the ground up

Five interactive slide decks explaining Monte Carlo sampling methods for people who know physics but
not statistics, plus an introduction page that ties them together.

Everything is plain HTML, CSS and JavaScript. There is no build step, no framework and no network
dependency: equations are pre-rendered text (KaTeX, with the fonts embedded in `assets/deck.css`),
the interactive demonstrations are written in vanilla JavaScript, and the three animations are MP4
files in `assets/`.

## Contents

| File | What it covers |
|---|---|
| `index.html` | Introduction to MCMC: the problem, Markov chains, detailed balance, correlation and cost, the three test problems, the summary table, and links to every deck |
| `metropolis.html` | **Random-walk Metropolis**: the algorithm, detailed balance, autocorrelation and effective sample size, the 0.234 rule, `O(d²)` scaling, relatives (MALA, slice, tempering) |
| `gibbs.html` | **Gibbs sampling**: conditionals, why nothing is rejected, axis-aligned geometry, `r₁ = ρ²`, hierarchical models, Gibbs-within-Metropolis |
| `hmc.html` | **Hamiltonian Monte Carlo**: momentum, Hamilton's equations, leapfrog and symplecticity, the Metropolis correction, tuning, resonance |
| `nuts.html` | **The No-U-Turn Sampler**: the U-turn criterion, doubling trees, why doubling keeps it exact, step-size adaptation, divergences |
| `nested.html` | **Nested sampling**: the evidence integral, prior volume and the shrinkage law, live points, weights, constrained draws, multimodality |

Each deck has an interactive demonstration you can play with:

- Metropolis — step-size slider, live acceptance rate and efficiency
- Gibbs — correlation slider, measured autocorrelation against theory
- HMC — Euler versus leapfrog in phase space; HMC versus a random walk on a 20:1 ridge
- NUTS — trajectory doubling until the U-turn, with the tree depth chosen each iteration
- Nested sampling — live points shrinking onto four peaks, with log *Z* accumulating

## Using the decks

- **← →** or swipe: move between slides
- **N**: show or hide the speaker notes
- **A− / A+**: change the text size
- **⌂ Contents**: back to the deck's title slide; **↩ All methods**: back to `index.html`

If JavaScript is unavailable, each deck degrades gracefully into one long scrolling page with every
slide and all the notes visible.

## Hosting on GitHub Pages

1. Create a repository and copy this folder's contents into it (or the folder itself).
2. Push to GitHub.
3. In the repository: **Settings → Pages → Build and deployment**, source **Deploy from a branch**,
   branch `main`, folder `/ (root)` (or `/docs` if you put the files there).
4. The site appears at `https://<username>.github.io/<repository>/` within a minute or two.

The file `.nojekyll` is included so that GitHub Pages serves every file as it is.

Nothing here needs a server: opening `index.html` from disk works just as well, which makes it easy
to check before publishing.

## Where the numbers come from

The tables and animations come from a study in which five samplers — random-walk Metropolis, Gibbs,
HMC, NUTS and PolyChord-style nested sampling — were run on three targets whose answers are known in
closed form:

1. a one-dimensional Gaussian posterior, exact answer `N(1.600, 0.447²)`;
2. a two-dimensional ridge with an aspect ratio of exactly 20;
3. a mixture of four Gaussians whose modes are 8.9 standard deviations apart.

Because the answers are known, the decks report **error**, not convergence diagnostics. The headline
result is that no method wins twice: gradients are decisive on the ridge and actively harmful when
the modes are separated.

## Credits

Prepared by Dipak Munshi with Claude (Anthropic), September 2026.
Equation rendering by [KaTeX](https://katex.org) (MIT licence); its CSS and fonts are embedded in
`assets/deck.css`.
