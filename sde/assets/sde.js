/* ---------------------------------------------------------------------------
   Stochastic differential equations in the browser: Brownian paths, the
   standard integrators, the Ito/Stratonovich distinction, ensemble densities,
   Feynman-Kac averaging and Langevin sampling.

   Everything is plain JavaScript and every number on the pages is computed at
   the moment it is displayed.
   --------------------------------------------------------------------------- */
var SDE = (function () {

  /* =======================================================================
     Random numbers.  Reproducible, so a demonstration can be reset to the
     identical realisation, and so that two schemes can be driven by the
     *same* Brownian path -- which is what makes a strong-error comparison
     meaningful at all.
     ======================================================================= */
  function Rng(seed) {
    var s = (seed >>> 0) || 2463534242;
    this.u = function () {                       /* xorshift32 */
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
    var spare = null;
    this.n = function () {                       /* Box-Muller, cached pair */
      if (spare !== null) { var v = spare; spare = null; return v; }
      var a = this.u() || 1e-12, b = this.u();
      var r = Math.sqrt(-2 * Math.log(a));
      spare = r * Math.sin(2 * Math.PI * b);
      return r * Math.cos(2 * Math.PI * b);
    };
  }

  /* A Brownian path on a fine grid: increments dW_k ~ N(0, dt).
     Coarser schemes sum consecutive increments, so every scheme sees the same
     underlying realisation of the noise. */
  function brownian(n, dt, rng) {
    var dW = new Float64Array(n), s = Math.sqrt(dt);
    for (var i = 0; i < n; i++) dW[i] = s * rng.n();
    return dW;
  }
  function cumsum(dW) {
    var W = new Float64Array(dW.length + 1), i;
    for (i = 0; i < dW.length; i++) W[i + 1] = W[i] + dW[i];
    return W;
  }

  /* =======================================================================
     Problems.  Each carries drift a(x,t), diffusion b(x,t), db/dx (needed by
     Milstein and by the Ito-Stratonovich conversion), and, where one exists,
     the exact solution as a function of the driving Brownian path.
     ======================================================================= */

  /* Geometric Brownian motion  dX = mu X dt + sigma X dW.
     Ito solution:          X(t) = X0 exp[(mu - sigma^2/2) t + sigma W(t)]
     Stratonovich solution: X(t) = X0 exp[ mu t + sigma W(t) ]
     The gap between them is the whole Ito/Stratonovich argument, and it is
     not a subtlety: at sigma = 1 it is a factor of e^{t/2}. */
  function gbm(o) {
    o = o || {};
    var mu = o.mu === undefined ? 1 : o.mu, sig = o.sigma === undefined ? 1 : o.sigma,
        x0 = o.x0 === undefined ? 1 : o.x0;
    return {
      name: 'geometric Brownian motion', x0: x0, mu: mu, sigma: sig,
      a: function (x) { return mu * x; },
      b: function (x) { return sig * x; },
      db: function () { return sig; },
      exact: function (t, W) { return x0 * Math.exp((mu - 0.5 * sig * sig) * t + sig * W); },
      exactStrat: function (t, W) { return x0 * Math.exp(mu * t + sig * W); },
      mean: function (t) { return x0 * Math.exp(mu * t); },
      hasExact: true
    };
  }

  /* Ornstein-Uhlenbeck  dX = -theta (X - m) dt + sigma dW.
     The physicist's SDE: a particle in a harmonic trap with thermal kicks.
     Stationary distribution N(m, sigma^2 / 2 theta) -- which is the
     fluctuation-dissipation relation in its simplest possible form. */
  function ou(o) {
    o = o || {};
    var th = o.theta === undefined ? 1 : o.theta, sig = o.sigma === undefined ? 1 : o.sigma,
        m = o.m === undefined ? 0 : o.m, x0 = o.x0 === undefined ? 2 : o.x0;
    return {
      name: 'Ornstein-Uhlenbeck', x0: x0, theta: th, sigma: sig, m: m,
      a: function (x) { return -th * (x - m); },
      b: function () { return sig; },
      db: function () { return 0; },              /* additive noise */
      meanT: function (t) { return m + (x0 - m) * Math.exp(-th * t); },
      varT: function (t) { return sig * sig / (2 * th) * (1 - Math.exp(-2 * th * t)); },
      statVar: sig * sig / (2 * th),
      density: function (x, t) {
        var mu = this.meanT(t), v = this.varT(t);
        return Math.exp(-(x - mu) * (x - mu) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
      },
      hasExact: false
    };
  }

  /* Overdamped Langevin in a potential U:  dX = -U'(X) dt + sqrt(2T) dW.
     Stationary density proportional to exp(-U/T) -- Boltzmann. */
  function langevin(U, dU, T, x0) {
    var temp = T === undefined ? 1 : T;
    return {
      name: 'Langevin', x0: x0 === undefined ? -1 : x0, T: temp, U: U, dU: dU,
      a: function (x) { return -dU(x); },
      b: function () { return Math.sqrt(2 * temp); },
      db: function () { return 0; },
      boltzmann: function (x) { return Math.exp(-U(x) / temp); },
      hasExact: false
    };
  }

  /* The standard symmetric double well, U = (x^2-1)^2 / 4.
     Minima at +-1, barrier height 1/4 at the origin: the Kramers problem. */
  function doubleWell(T, x0) {
    return langevin(function (x) { var y = x * x - 1; return 0.25 * y * y; },
                    function (x) { return x * (x * x - 1); }, T, x0);
  }

  /* =======================================================================
     Integrators.  Each advances one step given the Brownian increment.
     ======================================================================= */
  var schemes = {
    /* Euler-Maruyama: strong order 1/2 with multiplicative noise, weak order 1. */
    em: function (P, x, dt, dW) { return x + P.a(x) * dt + P.b(x) * dW; },

    /* Milstein: adds the term the Ito-Taylor expansion says is missing.
       Strong order 1.  Identical to Euler when b' = 0, i.e. additive noise --
       which is why the distinction never shows up for Ornstein-Uhlenbeck. */
    milstein: function (P, x, dt, dW) {
      var b = P.b(x);
      return x + P.a(x) * dt + b * dW + 0.5 * b * P.db(x) * (dW * dW - dt);
    },

    /* Heun / stochastic trapezoid.  Converges to the STRATONOVICH solution of
       the same coefficients, which is the cleanest way to see that the two
       calculi are different answers to the same written equation. */
    heun: function (P, x, dt, dW) {
      var ax = P.a(x), bx = P.b(x);
      var xb = x + ax * dt + bx * dW;
      return x + 0.5 * (ax + P.a(xb)) * dt + 0.5 * (bx + P.b(xb)) * dW;
    },

    /* Euler-Maruyama applied to the Ito equation equivalent to the
       Stratonovich one with these coefficients: a -> a + (1/2) b b'. */
    emStrat: function (P, x, dt, dW) {
      return x + (P.a(x) + 0.5 * P.b(x) * P.db(x)) * dt + P.b(x) * dW;
    }
  };

  /* Integrate one path from a given set of fine increments, taking `skip`
     of them per step -- so the same Brownian path can drive any step size. */
  function path(P, scheme, dW, dtFine, skip, keep) {
    var f = schemes[scheme], x = P.x0, dt = dtFine * skip, n = Math.floor(dW.length / skip);
    var out = keep ? [x] : null, i, k, inc;
    for (i = 0; i < n; i++) {
      inc = 0;
      for (k = 0; k < skip; k++) inc += dW[i * skip + k];
      x = f(P, x, dt, inc);
      if (keep) out.push(x);
    }
    return keep ? out : x;
  }

  /* =======================================================================
     Convergence.  Strong error is a pathwise comparison and needs the same
     noise for both; weak error compares distributions and does not.
     ======================================================================= */
  function strongError(P, scheme, T, nFine, skips, nPaths, seed) {
    var rng = new Rng(seed || 11), dtFine = T / nFine, res = [], i, j;
    var errs = new Float64Array(skips.length);
    for (j = 0; j < nPaths; j++) {
      var dW = brownian(nFine, dtFine, rng);
      var W = 0; for (i = 0; i < nFine; i++) W += dW[i];
      var ref = P.hasExact ? P.exact(T, W) : path(P, 'milstein', dW, dtFine, 1, false);
      for (i = 0; i < skips.length; i++)
        errs[i] += Math.abs(path(P, scheme, dW, dtFine, skips[i], false) - ref);
    }
    for (i = 0; i < skips.length; i++) res.push({dt: dtFine * skips[i], err: errs[i] / nPaths});
    return res;
  }

  function weakError(P, scheme, T, nFine, skips, nPaths, seed, f) {
    f = f || function (x) { return x; };
    var rng = new Rng(seed || 23), dtFine = T / nFine, res = [], i, j;
    var sums = new Float64Array(skips.length), ref = 0;
    for (j = 0; j < nPaths; j++) {
      var dW = brownian(nFine, dtFine, rng);
      var W = 0; for (i = 0; i < nFine; i++) W += dW[i];
      ref += f(P.hasExact ? P.exact(T, W) : path(P, 'milstein', dW, dtFine, 1, false));
      for (i = 0; i < skips.length; i++)
        sums[i] += f(path(P, scheme, dW, dtFine, skips[i], false));
    }
    ref /= nPaths;
    for (i = 0; i < skips.length; i++)
      res.push({dt: dtFine * skips[i], err: Math.abs(sums[i] / nPaths - ref)});
    return res;
  }

  /* Least-squares slope of log(err) against log(dt): the observed order. */
  function order(res) {
    var n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (var i = 0; i < res.length; i++) {
      if (!(res[i].err > 0)) continue;
      var x = Math.log(res[i].dt), y = Math.log(res[i].err);
      n++; sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    return n > 1 ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : NaN;
  }

  /* =======================================================================
     Ensembles: many paths at once, for densities and first-passage times.
     ======================================================================= */
  function Ensemble(P, scheme, dt, n, seed) {
    this.P = P; this.scheme = schemes[scheme] || schemes.em; this.dt = dt;
    this.rng = new Rng(seed || 7); this.t = 0;
    this.x = new Float64Array(n);
    for (var i = 0; i < n; i++) this.x[i] = P.x0;
    this.sq = Math.sqrt(dt);
  }
  Ensemble.prototype.step = function (k) {
    k = k || 1;
    for (var s = 0; s < k; s++) {
      for (var i = 0; i < this.x.length; i++)
        this.x[i] = this.scheme(this.P, this.x[i], this.dt, this.sq * this.rng.n());
      this.t += this.dt;
    }
  };
  Ensemble.prototype.hist = function (lo, hi, bins) {
    var h = new Float64Array(bins), w = (hi - lo) / bins, i, k, n = 0;
    for (i = 0; i < this.x.length; i++) {
      k = Math.floor((this.x[i] - lo) / w);
      if (k >= 0 && k < bins) { h[k]++; n++; }
    }
    for (i = 0; i < bins; i++) h[i] /= Math.max(1, this.x.length) * w;   /* density */
    return {h: h, w: w, lo: lo, inRange: n};
  };
  Ensemble.prototype.moments = function () {
    var n = this.x.length, m = 0, v = 0, i;
    for (i = 0; i < n; i++) m += this.x[i]; m /= n;
    for (i = 0; i < n; i++) v += (this.x[i] - m) * (this.x[i] - m);
    return {mean: m, varc: v / Math.max(1, n - 1)};
  };
  /* fraction of the ensemble in the right-hand well: the Kramers observable */
  Ensemble.prototype.fracRight = function () {
    var c = 0; for (var i = 0; i < this.x.length; i++) if (this.x[i] > 0) c++;
    return c / this.x.length;
  };

  /* =======================================================================
     Feynman-Kac.  For  du/dt = (1/2) sigma^2 u_xx - V(x) u  (imaginary time),
        u(x,T) = E[ f(X_T) exp(-int_0^T V(X_s) ds) ],  X a Brownian path from x.
     Averaging weighted random paths solves the PDE -- and for large T the
     weighted ensemble converges to the ground state of H = -(1/2)d2/dx2 + V,
     with the decay rate of the total weight giving E0.
     ======================================================================= */
  function feynmanKac(V, x0, T, dt, nPaths, sigma, seed, f) {
    sigma = sigma === undefined ? 1 : sigma;
    f = f || function () { return 1; };
    var rng = new Rng(seed || 5), nStep = Math.round(T / dt), s = sigma * Math.sqrt(dt);
    var sum = 0, sum2 = 0, i, k;
    for (i = 0; i < nPaths; i++) {
      var x = x0, acc = 0;
      for (k = 0; k < nStep; k++) {
        acc += V(x) * dt;                      /* left-point rule, order dt */
        x += s * rng.n();
      }
      var w = f(x) * Math.exp(-acc);
      sum += w; sum2 += w * w;
    }
    var m = sum / nPaths;
    return {value: m, err: Math.sqrt(Math.max(0, sum2 / nPaths - m * m) / nPaths)};
  }

  /* =======================================================================
     Metropolis-adjusted Langevin (MALA): the Langevin equation turned into an
     exact sampler by adding an accept/reject step, which removes the
     discretisation bias that plain Euler-Maruyama leaves behind.
     ======================================================================= */
  function Mala(U, dU, T, dt, x0, seed, adjust) {
    this.U = U; this.dU = dU; this.T = T; this.dt = dt; this.x = x0;
    this.rng = new Rng(seed || 17); this.acc = 0; this.n = 0; this.adjust = adjust !== false;
  }
  Mala.prototype.step = function (k) {
    k = k || 1;
    for (var s = 0; s < k; s++) {
      var dt = this.dt, T = this.T, x = this.x;
      var prop = x - this.dU(x) * dt + Math.sqrt(2 * T * dt) * this.rng.n();
      this.n++;
      if (!this.adjust) { this.x = prop; this.acc++; continue; }
      /* q(y|x) is Gaussian with mean x - U'(x) dt and variance 2 T dt */
      var f = function (from, to, self) {
        var m = from - self.dU(from) * dt, v = 2 * T * dt;
        return -(to - m) * (to - m) / (2 * v);
      };
      var logA = (-(this.U(prop) - this.U(x)) / T) + f(prop, x, this) - f(x, prop, this);
      if (Math.log(this.rng.u() || 1e-300) < logA) { this.x = prop; this.acc++; }
    }
    return this.x;
  };
  Mala.prototype.rate = function () { return this.n ? this.acc / this.n : 0; };

  function linspace(a, b, n) {
    var x = new Float64Array(n);
    for (var i = 0; i < n; i++) x[i] = a + (b - a) * i / (n - 1);
    return x;
  }

  return {
    Rng: Rng, brownian: brownian, cumsum: cumsum,
    gbm: gbm, ou: ou, langevin: langevin, doubleWell: doubleWell,
    schemes: schemes, path: path,
    strongError: strongError, weakError: weakError, order: order,
    Ensemble: Ensemble, feynmanKac: feynmanKac, Mala: Mala, linspace: linspace
  };
})();

if (typeof module !== 'undefined') module.exports = SDE;
