# Data Compression and AI — figure & note source

These are the scripts that generate every figure in the note
*Data Compression and AI* (https://dipakmunshi87.github.io/ML_Stat_Notes/compression_ai.html).
Each figure is produced by a small, self-contained Python script; the note HTML is
then assembled from those figures, with all mathematics pre-rendered to static SVG.

## Requirements

- Python 3.10+ with `numpy`, `matplotlib`, `scipy`, `pywavelets`, `torch`
  (`torch` is only needed for the convolutional autoencoder / VAE figures).
  ```
  pip install numpy matplotlib scipy pywavelets torch
  ```
- Node.js (for the maths pre-renderer), with `mathjax-full`:
  ```
  npm install mathjax-full
  ```

## Figure scripts (each writes a PNG into the current folder)

| script                | figure(s)                                              |
|-----------------------|--------------------------------------------------------|
| `make_compress.py`    | concept, results (PCA/wavelet/AE RMSE), weak-lensing maps |
| `make_1d.py`          | 1-D PCA; autoencoder-vs-PCA                             |
| `make_ae_appendix.py` | autoencoder by hand (1-D 4→1→4, 2-D 16→2→16)            |
| `make_appendixC.py`   | reconstruction vs realization (single mode, two modes, 2-D log-normal) |
| `make_vae.py`         | VAE: reconstruction vs prior samples                   |
| `make_reparam.py`     | the reparameterisation trick                           |
| `make_amortized.py`   | amortised inference (1-D)                               |
| `make_elbo.py`        | the ELBO of a 1-D linear-Gaussian VAE, worked by hand  |
| `make_flow.py`        | a normalizing flow in 1-D (x = e^z → log-normal)       |
| `make_ica.py`         | ICA = whitening (PCA) + a rotation                     |
| `make_preview.py`     | the 1200×630 social preview card                       |

## Building the note

1. Run the figure scripts above (any subset you want to regenerate).
2. `python build_compress_note.py` — embeds the figures (base64) and assembles the
   single scrolling note `compression_ai_note.html`.
3. `node prerender_math.js` — replaces all `$…$`, `$$…$$`, `\(…\)`, `\[…\]` with
   static SVG, so the maths renders with JavaScript disabled.
4. `python build_compress_deck.py` — paged (slide-deck) version.
5. `python split_publish.py` — splits into `compression_ai_part_A.html`
   (main note) + `compression_ai_part_B.html` (appendices) + the front page
   `compression_ai.html`, with cross-file deep links and the preview card.

Larger weak-lensing runs (higher-resolution convergence maps) use the
Imperial College Research Computing Service.

— Dr. Dipak Munshi, Advanced Research Fellow, Imperial College London
