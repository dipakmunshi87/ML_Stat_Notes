# The Physics of Bayesian Inference

Statistical mechanics, field theory and simulation-based inference, in thirty-eight chapters, a
summary and a historical appendix.

The course makes one claim and works it out: a posterior is a Boltzmann distribution, the evidence is
a partition function, the likelihood of a simulator is a path integral over its hidden variables, and
every estimator, from the Laplace approximation to neural posterior estimation and diffusion models,
is one of the strategies physicists developed for evaluating such integrals. It is written for people
who know physics and want to use, or understand, modern Bayesian and simulation-based methods.

**Read it:** open `index.html`, or https://dipakmunshi87.github.io/ML_Stat_Notes/bayesian-physics/

## What is in it

- **287 pages in 40 parts**, shown one page at a time, with a Contents button and a page counter.
- **A "connection" box at the top of every chapter**: the correspondence in one sentence and a small
  two-column table, inference in blue and physics in brick red.
- **Nine master tables** on the opening pages: variables, equations and methods, side by side.
- **117 worked examples**, each with an exact answer. Under each one a **Show the code** button opens
  a short Python program that reproduces its numbers, followed by the program's actual output.
- **Eleven experiments that run in the browser**: ABC on a curve fit, a diffusion model collapsing
  onto a posterior, a coin-toss phase transition with its Lee-Yang zeros, a particle jiggling in an
  energy well, a two-spin Boltzmann machine learning its coupling, stochastic gradient descent as a
  heat bath, and others.
- **Background and "Questions people ask"** in every chapter, a closing summary with open problems,
  and an appendix on how the two subjects have borrowed from each other since 1763.

## Using the page

- **Contents** (top bar): the list of chapters. **← →** buttons or arrow keys: previous and next page.
- **In this chapter**: the row of buttons at the top of each page lists the sections of the chapter.
- A particular page has its own address, for example `index.html#polyfit/7`.
- Everything is one self-contained HTML file with plain CSS and JavaScript. There is no build step and
  no framework. The only network request is for the IBM Plex fonts from Google Fonts; offline the page
  falls back to system fonts and otherwise works the same.

## Chapters and code

| | Chapter | Code for its worked examples | Examples |
|---|---|---|---|
| 1 | Start: the claim, nine master tables, and what you need to know first | – | – |
| 2 | The dictionary: inference as statistical physics | `code/02_physics.py` | 3 |
| 3 | Path integrals: the likelihood as a sum over histories | `code/03_path_integral.py` | 3 |
| 4 | Error bars are thermal fluctuations | `code/04_fluctuations.py` | 3 |
| 5 | Cumulants and correlation functions: describing what is not Gaussian | `code/05_correlators.py` | 4 |
| 6 | Maximum entropy: where distributions and priors come from | `code/06_maxent.py` | 4 |
| 7 | Information: how much a measurement tells you | `code/07_information.py` | 4 |
| 8 | Information geometry: the Fisher matrix is a metric | `code/08_geometry.py` | 3 |
| 9 | ABC: approximate Bayesian computation | `code/09_abc.py` | 4 |
| 10 | The curse of dimensionality | `code/10_curse.py` | 3 |
| 11 | Summaries, sufficiency and the Fisher limit | `code/11_summaries.py` | 4 |
| 12 | Eigenmodes: Karhunen-Loève, PCA and signal-to-noise | `code/12_eigenmodes.py` | 3 |
| 13 | Large deviations: the probability of the unlikely | `code/13_large_deviations.py` | 3 |
| 14 | Samplers: Metropolis, Hamiltonian and nested | `code/14_samplers.py` | 5 |
| 15 | Importance sampling and the work of changing a distribution | `code/15_importance.py` | 4 |
| 16 | Inference in time: filtering and the Kalman filter | `code/16_sequential.py` | 3 |
| 17 | Neural methods: NPE, NLE and NRE | `code/17_neural.py` | 4 |
| 18 | A worked example: fitting a line, a quadratic and a cubic | `code/18_polyfit.py` | 1 |
| 19 | Flows: reshaping a Gaussian into anything | `code/19_flows.py` | 5 |
| 20 | Amortisation: train once, infer many times | `code/20_amortisation.py` | 4 |
| 21 | Scores and diffusion models | `code/21_diffusion.py` | 5 |
| 22 | Energy-based models: learning the energy itself | `code/22_energy_based.py` | 3 |
| 23 | Training as physics: gradient descent with noise | `code/23_training.py` | 3 |
| 24 | Free energy: what training minimises | `code/24_free_energy.py` | 4 |
| 25 | Gaussian processes: when the unknown is a function | `code/25_gp.py` | 2 |
| 26 | The Wiener filter: a field answering a source | `code/26_wiener.py` | 3 |
| 27 | Nuisance parameters: integrating out what you do not want | `code/27_marginalisation.py` | 3 |
| 28 | Model comparison: evidence, Bayes factors and Occam's razor | `code/28_model_comparison.py` | 3 |
| 29 | Computing the evidence: thermodynamic integration and information criteria | `code/29_evidence.py` | 3 |
| 30 | Three ensembles, three kinds of inference | `code/30_ensembles.py` | 1 |
| 31 | Phase transitions in inference | `code/31_phase_transitions.py` | 3 |
| 32 | Symmetry and degeneracy: what the data cannot tell you | `code/32_symmetry.py` | 3 |
| 33 | Hierarchical models: many objects, one population | `code/33_hierarchical.py` | 2 |
| 34 | Hyperparameters: parameters about your assumptions | `code/34_hyperparameters.py` | 3 |
| 35 | Bias: three different things with one name | `code/35_bias.py` | 3 |
| 36 | Misspecification: when the simulator is wrong | `code/36_misspecification.py` | 2 |
| 37 | Checks: how to tell when the answer is wrong | `code/37_checks.py` | 3 |
| 38 | Applications: where SBI is used in physics | `code/38_applications.py` | 1 |
| 39 | Summary and open problems | – | – |
| 40 | Appendix: a shared history | – | – |

## Running the code

```
pip install numpy scipy
python code/14_samplers.py        # the worked examples of one chapter
python code/run_all.py            # all 117, about ten seconds
```

Each file defines one function per worked example and runs them all when executed. The programs are
the same ones shown under **Show the code** in the page, where their output is printed beside them.
The output in the page is inserted by the script that assembles it, from an actual run, so the text,
the code and the numbers were checked against each other.

`code/curve_fit_sbi/` holds the scripts and results behind the trained-network tables of chapter 18
(NPE, NLE, NRE, score-based and flow-matching estimators from the `sbi` package). They need PyTorch
and `sbi` and take tens of minutes. See the README in that folder.

## Where this lives

This folder is one topic of the ML_Stat_Notes site, served at
`https://dipakmunshi87.github.io/ML_Stat_Notes/bayesian-physics/`. It is self-contained: one HTML file
and a `code/` folder. The link "All topics" in its top bar returns to the master page.

## What to trust, and what to check

- **Numbers in worked examples** are reproduced by the code in `code/`, which runs without error and
  matches the text.
- **Tables from trained networks** (chapter 18) come from two random seeds on one small problem. The
  chapter says so, and they are not a general ranking of methods.
- **Names and dates** in the background sections and the historical appendix are given as usually
  cited and have not each been checked against the original publications. Corrections are welcome.
- **The correspondence itself** is exact where the text says "identity" and looser where it says
  "picture". The summary chapter lists where it stops.

## Credits

Prepared by Dipak Munshi with Claude (Anthropic), November 2026.
