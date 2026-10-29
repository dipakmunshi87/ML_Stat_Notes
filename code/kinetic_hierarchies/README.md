# Kinetic Hierarchies: from Liouville to Euler — code

Code for the note
[Kinetic Hierarchies: from Liouville to Euler](https://dipakmunshi87.github.io/ML_Stat_Notes/kinetic_hierarchies.html)
(Dr Dipak Munshi, ML & Statistics Notes). Every figure and animation in the note can be regenerated from this folder.

## Setup

```bash
pip install -r requirements.txt      # numpy, matplotlib, torch (CPU is fine)
```

All scripts run on a laptop CPU. Outputs go to `figures/`.

## Physics-informed neural networks (`pinn/`, Section 11)

| Script | What it does | Runtime (CPU) | Result in the note |
|---|---|---|---|
| `burgers_inverse.py` | Adhesion model (viscous Burgers). Learns the viscosity ν from 200 noisy samples; exact solution by Cole–Hopf | ~1 min | ν = 0.0299 (true 0.030), 0.18% error |
| `grad_shafranov_inverse.py` | Solov'ev tokamak equilibrium. Reconstructs the pressure-gradient constant from boundary flux and 40 noisy interior samples | ~1.5 min | c = 1.800 (true 1.8), ψ error 6×10⁻⁴ |
| `alfven_inverse.py` | Damped standing Alfvén wave. Learns the Alfvén speed and magnetic diffusivity from 300 noisy samples of u and b | ~3–5 min | v_A = 0.9997 (true 1.0), η = 0.0993 (true 0.1) |
| `hartmann_inverse.py` | Hartmann channel flow. Learns the Hartmann number from 25 noisy velocity samples | ~10 s | Ha = 6.01 (true 6) |
| `vlasov_poisson_skeleton.py` | The 1D-1V Vlasov–Poisson PINN shown in the note (residuals + quadrature for Gauss's law); a starting point, not a converged run | seconds | — |

`_plotstyle.py` holds the shared figure style. Each script writes `figures/<name>.png`, `.svg` and a `.json` with
the learned parameters and training history.

## Simulations (`sims/`, Sections 7, 9, 10 and 12)

Python versions of the five live animations (the browser versions are in `browser/`):

| Script | Physics | Note section |
|---|---|---|
| `two_stream_pic.py` | 1D electrostatic particle-in-cell, two-stream instability; prints the measured growth rate | 9 |
| `dynamical_friction.py` | Restricted N-body: heavy body sinking through a star field, softening 0.02 vs 0.5 | 7 (case three) |
| `orszag_tang.py` | 2D ideal MHD Orszag–Tang vortex; ∇·B = 0 exactly via a vector potential (`--n` sets the grid) | 9 |
| `orszag_tang_3d.py` | 3D resistive MHD Orszag–Tang vortex; div B = 0 to round-off via a vector potential; `--n` sets the grid (64³ ≈ 2 min) | 9 |
| `eft_1d.py` | Exact 1D cold-dark-matter sheets vs Zel'dovich; fits the EFT counterterm and compares n-point and field-level estimates of c_s² | 12 |
| `eft_2d_field_level.py` | 2D particle-mesh cold dark matter vs Zel'dovich and 2LPT forward models; field-level counterterm fit, error power, n-point vs field-level precision | 12.4b |
| `cmb_acoustic_toy.py` | Tight-coupling acoustic oscillations with Silk damping; odd/even peaks vs baryon loading R | 10 |

## C++ / CMake example (`cpp/`)

A self-contained CMake C++17 project for laptops and HPC: the two-stream PIC code with OpenMP threads, optional MPI,
CMake presets (`laptop`, `laptop-mpi`, `debug`, `hpc`), ctest physics tests (growth rate vs theory, momentum
conservation, stability) and a Slurm job script with a strong-scaling sweep. See `cpp/README.md`.

```bash
cd cpp && cmake --preset laptop && cmake --build --preset laptop -j && ctest --preset laptop
```

## Browser code (`browser/`)

`sims.js` (physics cores, no DOM, testable in node) and `anim.js` (canvas players) are the exact code embedded in
the note's HTML. To test a core in node: `node -e "const S=require('./sims.js'); const t=new S.TwoStream(); for(let i=0;i<200;i++) t.step(); console.log(t.WE)"`.

## Notes and conventions

* Units: plasma units (ω_p = 1) for the PIC run; G = M = a = 1 (Plummer) for the friction run; the MHD runs use the
  standard Orszag–Tang normalisation (γ = 5/3, ρ = γ², p = γ).
* The dynamical-friction run is a *restricted* N-body calculation: stars move in a fixed potential and feel the body;
  the body feels every star. This isolates Chandrasekhar's drag cleanly.
* The CMB animation is a pedagogical toy built on the matter-era tight-coupling solution (Hu & Sugiyama 1995), not a
  Boltzmann-code calculation; use CAMB or CLASS for real spectra.
* Random seeds are fixed, so reruns reproduce the figures; small differences between the Python and browser runs come
  from different random draws.

Licence: as for the rest of the ML_Stat_Notes repository.
