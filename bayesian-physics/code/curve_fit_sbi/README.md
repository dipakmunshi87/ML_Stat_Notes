# Trained-network tables of the curve-fitting chapter

These three scripts produced the tables in chapter 18, "A worked example: fitting a line, a quadratic
and a cubic". They are research scripts, kept as they were run, and are slower than the worked
examples.

| Script | What it does | Needs | Run time (CPU) |
|---|---|---|---|
| `exact_abc.py` | Makes the 20 data points, the exact posteriors and evidences, and the ABC acceptance table from 4 million simulations per model. Writes `exact_abc.json`. | numpy | about 1 minute |
| `neural.py` | Trains NPE, NLE and NRE (the `sbi` package, default networks) on 5,000 simulations, for the three models and two seeds. Reads `exact_abc.json`, writes `neural.json`. | torch, sbi | about 20 minutes |
| `diffusion.py` | The same for score-based (NPSE) and flow-matching (FMPE) posterior estimation. Writes `diffusion.json`. | torch, sbi | about 10 minutes |

Run `exact_abc.py` first. The `.json` files here are the outputs behind the numbers in the text.
`neural50k.json` is the single check with 50,000 simulations (NPE and NLE on the quadratic, one seed).

Results from trained networks depend on the random seed, the package version and the hardware, so a
rerun will reproduce the tables to within the seed-to-seed scatter they report, not digit for digit.
The versions used were sbi 0.27.0, PyTorch 2.15.0, NumPy 2.5.3, SciPy 1.18.1, on CPU.
