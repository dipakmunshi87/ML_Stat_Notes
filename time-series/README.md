# Time series for physicists — an illustrated introduction

Eight interactive pages defining time-series analysis, translating its vocabulary into physics, setting out
the standard models and how they are fitted, and working through four examples from modern physics.

Plain HTML, CSS and JavaScript: no build step, no framework, no network dependency. Equations are
pre-rendered text (KaTeX, fonts embedded in `assets/deck.css`), and every figure is **computed live** in
the browser rather than being an image, so each reload gives a new realisation.

## Contents

| File | What it covers |
|---|---|
| `index.html` | Front page: what a time series is, the correlation function and power spectrum, a short physics dictionary, five habits worth having, and links to every page |
| `basics.html` | **Definitions**: sampling, stationarity, ergodicity, autocorrelation, power spectral density, correlation time, Nyquist, aliasing, spectral leakage, vocabulary table |
| `dictionary.html` | **Physics dictionary**: term-by-term translation; AR(1) shown to be the Ornstein–Uhlenbeck process sampled; filters as Green's functions; four places the fields disagree |
| `models.html` | **Families and models**: white noise, AR, MA, ARMA, ARIMA, random walk, periodic, 1/f red noise, changing variance, state-space, Gaussian processes; how to tell them apart |
| `methods.html` | **Practice**: moment estimators, Yule–Walker, likelihood via the Kalman filter, periodograms, windows, Welch averaging, uneven sampling and Lomb–Scargle, filtering, whitening, residual tests |
| `examples.html` | **Four worked examples**: gravitational-wave matched filtering; pulsar-timing red noise; quasar variability as an OU process; satellite radio interference in time–frequency data |
| `applications.html` | **Eight more physics problems**: asteroseismology, exoplanet transits with correlated noise, X-ray quasi-periodic oscillations, CMB detector drifts, ambient-noise seismology, atomic clocks and the Allan variance, 1/f and telegraph noise in devices, solar-wind turbulence — plus a table of eight further fields |
| `errors.html` | **Error estimates, limitations and extensions**: effective sample size, Bartlett's formula, χ² statistics of spectra, biases, trials factors and red-noise significance, block bootstrap, surrogate data and injection tests, and a table of assumptions with the standard way each is lifted |
| `software.html` | **Software and reading list**: public open-source tools with links (scipy, statsmodels, astropy, celerite2, Stingray, GWpy, PyCBC, Bilby, PINT, enterprise, ObsPy, allantools, …), textbooks and the key papers |

## The interactive figures

- **basics** — one memory parameter φ, shown at once as a series, an autocorrelation and a spectrum, with the
  theory drawn on top
- **dictionary** — a single Ornstein–Uhlenbeck trajectory sampled at three cadences; the measured AR(1)
  coefficient matches `exp(−Δt/τ)` in each panel
- **models** — a gallery of nine families, each shown as series, autocorrelation and spectrum
- **methods** — a spectrum estimator you can break: window on or off, Welch averaging on or off, signal
  amplitude adjustable
- **examples** — a chirp buried in red noise, recovered by matched filtering, with whitening you can switch off
- **errors** — how noisy a spectral estimate really is: the χ²₂ₖ distribution of Ŝ/S, and the 1/√K gain from averaging

## Using the pages

- **← →** or swipe: move between slides
- **N**: show or hide the speaker notes
- **A− / A+**: change the text size
- **⌂ Contents**: back to the page's title slide; **↩ All pages**: back to `index.html`

With JavaScript unavailable, each page degrades into one long scrolling document with every slide and all the
notes visible (the live figures need JavaScript).

## Hosting on GitHub Pages

1. Copy this folder's contents into a repository and push.
2. **Settings → Pages → Build and deployment**, source **Deploy from a branch**, branch `main`, folder
   `/ (root)`.
3. The site appears at `https://<username>.github.io/<repository>/`.

`.nojekyll` is included so GitHub Pages serves every file untouched. Nothing needs a server: opening
`index.html` from disk works identically.

## Credits

Prepared by Dipak Munshi with Claude (Anthropic), September 2026.
Equation rendering by [KaTeX](https://katex.org) (MIT licence). Companion set: **MCMC demo set** — five
interactive decks on Monte Carlo samplers.
