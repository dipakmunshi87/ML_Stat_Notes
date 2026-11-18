// Physics cores for the page animations. Pure functions/classes: no DOM, so they can be tested in node.
(function (root) {
  // small seeded RNG so every reset reproduces the same run
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

  // ---------------- 1. two-stream instability: 1D-1V electrostatic PIC ----------------
  // plasma units (omega_p = 1, lambda scaled), electrons on a fixed neutralising ion background
  class TwoStream {
    constructor(o = {}) {
      this.N = o.N || 20000; this.Ng = o.Ng || 64; this.L = o.L || 4 * Math.PI;
      this.v0 = o.v0 || 1.0; this.vth = o.vth ?? 0.05; this.dt = o.dt || 0.1; this.reset();
    }
    reset() {
      const r = rng(7), N = this.N, L = this.L;
      this.x = new Float64Array(N); this.v = new Float64Array(N); this.beam = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        const b = i % 2; this.beam[i] = b;
        let x = (Math.floor(i / 2) + 0.5) * L / (N / 2);
        x += 1e-3 * L * Math.cos(2 * Math.PI * x / L);              // seed the fundamental mode
        this.x[i] = ((x % L) + L) % L;
        this.v[i] = (b ? 1 : -1) * this.v0 + this.vth * gauss(r);
      }
      this.t = 0; this.Ehist = []; this.field(); this.halfKick(-1);
    }
    field() {
      const Ng = this.Ng, dx = this.L / Ng, rho = new Float64Array(Ng), N = this.N;
      for (let i = 0; i < N; i++) {                                  // cloud-in-cell deposit
        const g = this.x[i] / dx - 0.5, j = Math.floor(g), f = g - j;
        rho[(j + Ng) % Ng] += 1 - f; rho[(j + 1) % Ng] += f;
      }
      const n0 = N / Ng; const E = new Float64Array(Ng); let acc = 0, mean = 0;
      for (let j = 0; j < Ng; j++) { acc += (1 - rho[j] / n0) * dx; E[j] = acc; mean += acc; }   // dE/dx = 1 - n_e
      mean /= Ng; let W = 0;
      for (let j = 0; j < Ng; j++) { E[j] -= mean; W += 0.5 * E[j] * E[j] * dx; }
      this.E = E; this.rho = rho; this.WE = W;
    }
    Eat(x) { const dx = this.L / this.Ng, Ng = this.Ng, g = x / dx - 0.5, j = Math.floor(g), f = g - j;
      return (1 - f) * this.E[(j + Ng) % Ng] + f * this.E[(j + 1) % Ng]; }
    halfKick(s) { for (let i = 0; i < this.N; i++) this.v[i] -= s * 0.5 * this.dt * this.Eat(this.x[i]); }
    step() {                                                         // leapfrog: v lives at half steps
      const N = this.N, L = this.L, dt = this.dt;
      for (let i = 0; i < N; i++) this.v[i] -= dt * this.Eat(this.x[i]);
      for (let i = 0; i < N; i++) { let x = this.x[i] + dt * this.v[i]; x %= L; if (x < 0) x += L; this.x[i] = x; }
      this.field(); this.t += dt; this.Ehist.push([this.t, this.WE]);
    }
  }

  // ---------------- 2. dynamical friction: restricted N-body ----------------
  // stars are tracers in a fixed Plummer potential (G = M = a = 1) and feel the heavy body;
  // the body feels every star through a softened force. Same initial stars for every softening.
  class Friction {
    constructor(o = {}) { this.N = o.N || 4000; this.Mb = o.Mb || 0.08; this.eps = o.eps ?? 0.05;
      this.r0 = o.r0 || 2.5; this.dt = o.dt || 0.01; this.seed = o.seed || 3; this.reset(); }
    reset() {
      const r = rng(this.seed), N = this.N;
      this.x = new Float64Array(N); this.y = new Float64Array(N); this.vx = new Float64Array(N); this.vy = new Float64Array(N);
      this.m = 1 / N;                                               // stars carry the halo's mass
      for (let i = 0; i < N; i++) {
        // 2D Plummer-like disc: radius from the projected Plummer profile, near-isotropic velocities
        const u = r(); const R = Math.sqrt(u / (1 - u)) ; const rr = Math.min(R, 8);
        const th = 2 * Math.PI * r();
        this.x[i] = rr * Math.cos(th); this.y[i] = rr * Math.sin(th);
        const vc = Math.sqrt(rr * rr / Math.pow(rr * rr + 1, 1.5));
        const s = 0.55 * vc + 0.12;                                  // hot: dispersion comparable to rotation
        const vt = 0.45 * vc + s * gauss(r), vr = s * gauss(r);
        this.vx[i] = vr * Math.cos(th) - vt * Math.sin(th); this.vy[i] = vr * Math.sin(th) + vt * Math.cos(th);
      }
      // let the tracers settle in the fixed potential before the body is added
      for (let s = 0; s < 1500; s++) for (let i = 0; i < N; i++) {
        const x = this.x[i], y = this.y[i], ih = 1 / Math.pow(x * x + y * y + 1, 1.5);
        this.vx[i] -= 0.01 * x * ih; this.vy[i] -= 0.01 * y * ih; this.x[i] += 0.01 * this.vx[i]; this.y[i] += 0.01 * this.vy[i]; }
      // the body feels only the stars (they ARE the halo): start it on the circular orbit of their mean field
      let gr = 0; const e2 = 0.09; // smooth estimate of the mean field, independent of the softening under test
      for (let a = 0; a < 32; a++) { const bx = this.r0 * Math.cos(a * Math.PI / 16), by = this.r0 * Math.sin(a * Math.PI / 16);
        for (let i = 0; i < N; i++) { const dx = bx - this.x[i], dy = by - this.y[i], d2 = dx * dx + dy * dy + e2;
          gr += this.m * (dx * bx + dy * by) / this.r0 / (d2 * Math.sqrt(d2)); } }
      gr /= 32;
      this.b = { x: this.r0, y: 0, vx: 0, vy: Math.sqrt(Math.max(gr, 1e-6) * this.r0) };
      this.t = 0; this.rhist = [[0, this.r0]];
    }
    step(nsub = 1) {
      const N = this.N, dt = this.dt, e2 = this.eps * this.eps, Mb = this.Mb, m = this.m, b = this.b;
      for (let s = 0; s < nsub; s++) {
        let ax = 0, ay = 0;
        for (let i = 0; i < N; i++) {
          const x = this.x[i], y = this.y[i];
          const r2 = x * x + y * y, ih = 1 / Math.pow(r2 + 1, 1.5);   // Plummer pull
          const dx = b.x - x, dy = b.y - y, d2 = dx * dx + dy * dy + e2, id = 1 / (d2 * Math.sqrt(d2));
          this.vx[i] += dt * (-x * ih + Mb * dx * id);
          this.vy[i] += dt * (-y * ih + Mb * dy * id);
          this.x[i] += dt * this.vx[i]; this.y[i] += dt * this.vy[i];
          ax -= m * dx * id; ay -= m * dy * id;
        }
        b.vx += dt * ax; b.vy += dt * ay;
        b.x += dt * b.vx; b.y += dt * b.vy; this.t += dt;
      }
      this.rhist.push([this.t, Math.hypot(b.x, b.y)]);
    }
  }

  // ---------------- 3. Orszag-Tang vortex: 2D ideal MHD ----------------
  // conserved (rho, mx, my, E) on a periodic grid with Rusanov fluxes + MUSCL/minmod, RK2;
  // B = curl(A z^) from a vector potential A advected with the flow, so div B = 0 exactly.
  class OrszagTang {
    constructor(o = {}) { this.n = o.n || 128; this.gam = 5 / 3; this.cfl = 0.4; this.reset(); }
    reset() {
      const n = this.n, L = 2 * Math.PI, h = L / n, g = this.gam; this.h = h;
      this.U = [0, 1, 2, 3].map(() => new Float64Array(n * n)); this.A = new Float64Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const x = (i + 0.5) * h, y = (j + 0.5) * h, k = j * n + i;
        const rho = g * g, p = g, vx = -Math.sin(y), vy = Math.sin(x);
        const bx = -Math.sin(y), by = Math.sin(2 * x);
        this.A[k] = Math.cos(y) + 0.5 * Math.cos(2 * x);
        this.U[0][k] = rho; this.U[1][k] = rho * vx; this.U[2][k] = rho * vy;
        this.U[3][k] = p / (g - 1) + 0.5 * rho * (vx * vx + vy * vy) + 0.5 * (bx * bx + by * by);
      }
      this.t = 0;
    }
    B(A) { // B_x = dA/dy, B_y = -dA/dx (central differences, cell-centred)
      const n = this.n, h = this.h, bx = new Float64Array(n * n), by = new Float64Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const k = j * n + i, ip = j * n + (i + 1) % n, im = j * n + (i + n - 1) % n,
          jp = ((j + 1) % n) * n + i, jm = ((j + n - 1) % n) * n + i;
        bx[k] = (A[jp] - A[jm]) / (2 * h); by[k] = -(A[ip] - A[im]) / (2 * h);
      }
      return [bx, by];
    }
    rhs(U, A) {
      const n = this.n, h = this.h, g = this.gam, [bx, by] = this.B(A);
      const dU = [0, 1, 2, 3].map(() => new Float64Array(n * n)), dA = new Float64Array(n * n);
      const prim = (k) => { const r = U[0][k], vx = U[1][k] / r, vy = U[2][k] / r, b2 = bx[k] * bx[k] + by[k] * by[k];
        const p = Math.max((g - 1) * (U[3][k] - 0.5 * r * (vx * vx + vy * vy) - 0.5 * b2), 1e-6); return [r, vx, vy, p]; };
      const P = new Array(n * n); for (let k = 0; k < n * n; k++) P[k] = prim(k);
      const mm = (a, b) => (a * b <= 0 ? 0 : (Math.abs(a) < Math.abs(b) ? a : b));
      let smax = 0;
      // flux in direction d (0: x, 1: y). B normal component is continuous: average of the two cells.
      const flux = (W, bn, bt, d) => { const [r, vx, vy, p] = W; const vn = d ? vy : vx, vt = d ? vx : vy;
        const b2 = bn * bn + bt * bt, pt = p + 0.5 * b2, E = p / (g - 1) + 0.5 * r * (vx * vx + vy * vy) + 0.5 * b2;
        const fn = r * vn * vn + pt - bn * bn, ft = r * vn * vt - bn * bt;
        return [r * vn, d ? ft : fn, d ? fn : ft, (E + pt) * vn - bn * (vx * bx_ + vy * by_)]; };
      let bx_, by_;
      for (let d = 0; d < 2; d++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const k = j * n + i, kp = d ? ((j + 1) % n) * n + i : j * n + (i + 1) % n,
          km = d ? ((j + n - 1) % n) * n + i : j * n + (i + n - 1) % n,
          kpp = d ? ((j + 2) % n) * n + i : j * n + (i + 2) % n;
        // reconstruct primitive states at face k+1/2
        const WL = [], WR = [];
        for (let q = 0; q < 4; q++) {
          WL.push(P[k][q] + 0.5 * mm(P[k][q] - P[km][q], P[kp][q] - P[k][q]));
          WR.push(P[kp][q] - 0.5 * mm(P[kp][q] - P[k][q], P[kpp][q] - P[kp][q]));
        }
        WL[0] = Math.max(WL[0], 1e-6); WR[0] = Math.max(WR[0], 1e-6); WL[3] = Math.max(WL[3], 1e-6); WR[3] = Math.max(WR[3], 1e-6);
        const bnL = d ? by[k] : bx[k], bnR = d ? by[kp] : bx[kp], bn = 0.5 * (bnL + bnR);
        const btL = d ? bx[k] : by[k], btR = d ? bx[kp] : by[kp];
        const cf = (W, bt) => { const b2 = bn * bn + bt * bt, a2 = g * W[3] / W[0], va2 = b2 / W[0];
          return Math.sqrt(0.5 * (a2 + va2 + Math.sqrt(Math.max((a2 + va2) ** 2 - 4 * a2 * bn * bn / W[0], 0)))); };
        const s = Math.max(Math.abs(d ? WL[2] : WL[1]) + cf(WL, btL), Math.abs(d ? WR[2] : WR[1]) + cf(WR, btR));
        smax = Math.max(smax, s);
        bx_ = d ? btL : bn; by_ = d ? bn : btL; const FL = flux(WL, bn, btL, d);
        const UL = [WL[0], WL[0] * WL[1], WL[0] * WL[2], WL[3] / (g - 1) + 0.5 * WL[0] * (WL[1] ** 2 + WL[2] ** 2) + 0.5 * (bn * bn + btL * btL)];
        bx_ = d ? btR : bn; by_ = d ? bn : btR; const FR = flux(WR, bn, btR, d);
        const UR = [WR[0], WR[0] * WR[1], WR[0] * WR[2], WR[3] / (g - 1) + 0.5 * WR[0] * (WR[1] ** 2 + WR[2] ** 2) + 0.5 * (bn * bn + btR * btR)];
        for (let q = 0; q < 4; q++) { const F = 0.5 * (FL[q] + FR[q]) - 0.5 * s * (UR[q] - UL[q]);
          dU[q][k] -= F / h; dU[q][kp] += F / h; }
      }
      // A is advected: dA/dt = -u . grad A, second-order upwind
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const k = j * n + i, vx = P[k][1], vy = P[k][2];
        const a = (o) => A[((j + Math.floor(o / n) + n) % n) * n + (i + (o % n) + n) % n];
        const Ax = vx > 0 ? (3 * A[k] - 4 * A[j * n + (i + n - 1) % n] + A[j * n + (i + n - 2) % n]) / (2 * h)
                          : (-3 * A[k] + 4 * A[j * n + (i + 1) % n] - A[j * n + (i + 2) % n]) / (2 * h);
        const Ay = vy > 0 ? (3 * A[k] - 4 * A[((j + n - 1) % n) * n + i] + A[((j + n - 2) % n) * n + i]) / (2 * h)
                          : (-3 * A[k] + 4 * A[((j + 1) % n) * n + i] - A[((j + 2) % n) * n + i]) / (2 * h);
        dA[k] = -(vx * Ax + vy * Ay);
      }
      return [dU, dA, smax];
    }
    step() {
      const n2 = this.n * this.n, U = this.U, A = this.A;
      const [d1, a1, s] = this.rhs(U, A); const dt = this.cfl * this.h / s;
      const U1 = U.map((u, q) => u.map((v, k) => v + dt * d1[q][k])), A1 = A.map((v, k) => v + dt * a1[k]);
      const [d2, a2] = this.rhs(U1, A1);
      for (let q = 0; q < 4; q++) for (let k = 0; k < n2; k++) U[q][k] = 0.5 * (U[q][k] + U1[q][k] + dt * d2[q][k]);
      for (let k = 0; k < n2; k++) A[k] = 0.5 * (A[k] + A1[k] + dt * a2[k]);
      this.t += dt; return dt;
    }
  }

  // ---------------- 4. CMB acoustic toy: tight coupling, then free streaming ----------------
  // Hu & Sugiyama-style matter-era approximation for the effective temperature on a 2D patch:
  //   [Theta_0 + Psi](k, eta) = Psi_k [ (1+3R)/3 cos(k r_s) - R ] exp(-k^2 / k_D^2),
  //   r_s = c_s eta, c_s = 1/sqrt(3(1+R)), 1/k_D^2 grows linearly in eta (photon random walk).
  function fft1(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inv ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const ar = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci, ai = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
          re[i + j + len / 2] = re[i + j] - ar; im[i + j + len / 2] = im[i + j] - ai; re[i + j] += ar; im[i + j] += ai;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
  }
  class CMB {
    constructor(o = {}) { this.n = o.n || 128; this.R = o.R ?? 0.6; this.kPeak = o.kPeak || 7; this.kDstar = o.kDstar || 30; this.reset(); }
    reset() {
      const n = this.n, r = rng(11); this.psi = []; this.kk = new Float64Array(n * n);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const kx = i <= n / 2 ? i : i - n, ky = j <= n / 2 ? j : j - n, k = Math.hypot(kx, ky);
        this.kk[j * n + i] = k;
        const amp = k > 0 ? 1 / k : 0;                                // scale-invariant-ish 2D potential
        this.psi.push([amp * gauss(r), amp * gauss(r)]);
      }
      this.eta = 0;
    }
    // k is an integer wavenumber on the patch; eta runs from 0 to 1 (= recombination).
    // sound horizon r_s(eta) = r_s* eta puts the first compression peak at k r_s* = pi, i.e. k1 = kPeak;
    // the Silk scale k_D(eta) = kD* / sqrt(eta) moves to larger scales as photons random-walk further.
    transfer(k, eta) {
      const R = this.R, rs = Math.PI / this.kPeak * eta, kD = this.kDstar / Math.sqrt(Math.max(eta, 1e-4));
      return ((1 + 3 * R) / 3 * Math.cos(k * rs) - R) * Math.exp(-(k * k) / (kD * kD));
    }
    map(eta) {
      const n = this.n, re = new Float64Array(n * n), im = new Float64Array(n * n);
      for (let q = 0; q < n * n; q++) { const T = this.transfer(this.kk[q], eta); re[q] = this.psi[q][0] * T; im[q] = this.psi[q][1] * T; }
      // 2D inverse FFT by rows then columns
      const rr = new Float64Array(n), ii = new Float64Array(n);
      for (let j = 0; j < n; j++) { for (let i = 0; i < n; i++) { rr[i] = re[j * n + i]; ii[i] = im[j * n + i]; }
        fft1(rr, ii, true); for (let i = 0; i < n; i++) { re[j * n + i] = rr[i]; im[j * n + i] = ii[i]; } }
      for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) { rr[j] = re[j * n + i]; ii[j] = im[j * n + i]; }
        fft1(rr, ii, true); for (let j = 0; j < n; j++) { re[j * n + i] = rr[j]; im[j * n + i] = ii[j]; } }
      return re;
    }
  }

  root.Sims = { TwoStream, Friction, OrszagTang, CMB, rng };
  if (typeof module !== 'undefined') module.exports = root.Sims;
})(typeof window !== 'undefined' ? window : globalThis);
