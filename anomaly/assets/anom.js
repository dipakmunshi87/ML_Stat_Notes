/* Anomaly-detection algorithms used by the demonstrations on these pages.
   Plain ES5, no dependencies beyond PL for colours. Everything is written for
   clarity first: the browser has a few hundred points to deal with, not a few
   million, so the O(n^2) routines are left as O(n^2) where that is what the
   textbook definition says. */
var AD = (function () {

  /* =================================================================
     0.  random numbers, reproducible
     ================================================================= */
  function RNG(seed) {
    var s = (seed >>> 0) || 20260925;
    this.u = function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    this.n = function () {
      var a = this.u() || 1e-12, b = this.u();
      return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
    };
    this.pick = function (arr) { return arr[Math.floor(this.u() * arr.length) % arr.length]; };
    this.int = function (n) { return Math.floor(this.u() * n) % n; };
  }

  /* =================================================================
     1.  univariate summaries
     ================================================================= */
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function sd(a) {
    var m = mean(a), s = 0, i;
    for (i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / Math.max(1, a.length - 1));
  }
  function quantile(a, p) {
    var b = a.slice().sort(function (x, y) { return x - y; });
    var h = (b.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
    return b[lo] + (h - lo) * (b[hi] - b[lo]);
  }
  function median(a) { return quantile(a, 0.5); }
  function mad(a) {
    var m = median(a), d = [], i;
    for (i = 0; i < a.length; i++) d.push(Math.abs(a[i] - m));
    return median(d);
  }

  /* classical z, and the robust replacement */
  function zscore(a) {
    var m = mean(a), s = sd(a) || 1e-12, o = [], i;
    for (i = 0; i < a.length; i++) o.push(Math.abs(a[i] - m) / s);
    return o;
  }
  function modz(a) {
    var m = median(a), d = mad(a) || 1e-12, o = [], i;
    for (i = 0; i < a.length; i++) o.push(0.6745 * Math.abs(a[i] - m) / d);
    return o;
  }
  function tukey(a, mult) {
    var q1 = quantile(a, 0.25), q3 = quantile(a, 0.75), iqr = q3 - q1, k = (mult === undefined ? 1.5 : mult);
    return { q1: q1, q3: q3, iqr: iqr, lo: q1 - k * iqr, hi: q3 + k * iqr };
  }

  /* =================================================================
     2.  two-dimensional Gaussian machinery
         (everything here is written for p = 2 so the algebra stays closed-form
          and visible; the generalisation is standard but not illuminating)
     ================================================================= */
  function meanCov(X, idx) {
    var n = idx ? idx.length : X.length, i, k, mx = 0, my = 0;
    for (i = 0; i < n; i++) { k = idx ? idx[i] : i; mx += X[k][0]; my += X[k][1]; }
    mx /= n; my /= n;
    var sxx = 0, syy = 0, sxy = 0, dx, dy;
    for (i = 0; i < n; i++) {
      k = idx ? idx[i] : i; dx = X[k][0] - mx; dy = X[k][1] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    var d = Math.max(1, n - 1);
    return { mu: [mx, my], S: [[sxx / d, sxy / d], [sxy / d, syy / d]] };
  }
  function inv2(S) {
    var det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (Math.abs(det) < 1e-14) det = det >= 0 ? 1e-14 : -1e-14;
    return { M: [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]], det: det };
  }
  function mahal2(X, mu, S) {
    var I = inv2(S).M, o = [], i, dx, dy;
    for (i = 0; i < X.length; i++) {
      dx = X[i][0] - mu[0]; dy = X[i][1] - mu[1];
      o.push(dx * (I[0][0] * dx + I[0][1] * dy) + dy * (I[1][0] * dx + I[1][1] * dy));
    }
    return o;
  }
  function mahalPoint(x, y, mu, S) {
    var I = inv2(S).M, dx = x - mu[0], dy = y - mu[1];
    return dx * (I[0][0] * dx + I[0][1] * dy) + dy * (I[1][0] * dx + I[1][1] * dy);
  }

  /* Minimum Covariance Determinant, concentration-step version (FAST-MCD
     without the nesting). Start from a subset, fit, keep the h points with the
     smallest Mahalanobis distance, refit, repeat. Several random starts; keep
     the fit with the smallest determinant. */
  function mcd2(X, frac, starts, seed) {
    var n = X.length, h = Math.max(4, Math.floor((frac || 0.75) * n));
    var rng = new RNG(seed || 7), best = null, s, it, i;
    for (s = 0; s < (starts || 8); s++) {
      var idx = [];
      while (idx.length < h) { var j = rng.int(n); if (idx.indexOf(j) < 0) idx.push(j); }
      var fit = meanCov(X, idx);
      for (it = 0; it < 12; it++) {
        var d = mahal2(X, fit.mu, fit.S), ord = [];
        for (i = 0; i < n; i++) ord.push([d[i], i]);
        ord.sort(function (a, b) { return a[0] - b[0]; });
        var nid = [];
        for (i = 0; i < h; i++) nid.push(ord[i][1]);
        var nf = meanCov(X, nid);
        if (Math.abs(inv2(nf.S).det - inv2(fit.S).det) < 1e-12) { fit = nf; break; }
        fit = nf;
      }
      var det = Math.abs(inv2(fit.S).det);
      if (!best || det < best.det) best = { mu: fit.mu, S: fit.S, det: det };
    }
    /* consistency factor so the scale matches the classical estimate on clean
       Gaussian data; the usual chi-square correction at the median */
    var dd = mahal2(X, best.mu, best.S), c = median(dd) / 1.3863;   /* chi2_2 median = 2 ln 2 */
    if (c > 1e-9) {
      best.S = [[best.S[0][0] * c, best.S[0][1] * c], [best.S[1][0] * c, best.S[1][1] * c]];
    }
    return best;
  }

  /* eigen-decomposition of a symmetric 2x2, for drawing ellipses */
  function eig2(S) {
    var a = S[0][0], b = S[0][1], d = S[1][1];
    var tr = a + d, det = a * d - b * b;
    var disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    var v1 = Math.abs(b) > 1e-12 ? [l1 - d, b] : [1, 0];
    var nrm = Math.hypot(v1[0], v1[1]) || 1;
    return { l: [Math.max(l1, 1e-12), Math.max(l2, 1e-12)], v: [v1[0] / nrm, v1[1] / nrm] };
  }
  /* points on the c-sigma ellipse of (mu,S) */
  function ellipse(mu, S, c, m) {
    var e = eig2(S), out = [], i, t, ax = Math.sqrt(e.l[0]) * c, ay = Math.sqrt(e.l[1]) * c;
    var cs = e.v[0], sn = e.v[1];
    m = m || 96;
    for (i = 0; i <= m; i++) {
      t = 2 * Math.PI * i / m;
      var px = ax * Math.cos(t), py = ay * Math.sin(t);
      out.push([mu[0] + cs * px - sn * py, mu[1] + sn * px + cs * py]);
    }
    return out;
  }

  /* =================================================================
     3.  distance and density
     ================================================================= */
  function distMatrix(X) {
    var n = X.length, D = [], i, j, s, k, p = X[0].length;
    for (i = 0; i < n; i++) D.push(new Float64Array(n));
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
      s = 0; for (k = 0; k < p; k++) { var d = X[i][k] - X[j][k]; s += d * d; }
      s = Math.sqrt(s); D[i][j] = s; D[j][i] = s;
    }
    return D;
  }
  /* distance to the kth nearest neighbour */
  function knnDist(X, k, D) {
    D = D || distMatrix(X);
    var n = X.length, o = [], i, row;
    for (i = 0; i < n; i++) {
      row = [];
      for (var j = 0; j < n; j++) if (j !== i) row.push(D[i][j]);
      row.sort(function (a, b) { return a - b; });
      o.push(row[Math.min(k, row.length) - 1]);
    }
    return o;
  }
  /* Local Outlier Factor, Breunig et al. 2000, written as defined */
  function lof(X, k, D) {
    D = D || distMatrix(X);
    var n = X.length, i, j, nb = [], kd = [];
    for (i = 0; i < n; i++) {
      var row = [];
      for (j = 0; j < n; j++) if (j !== i) row.push([D[i][j], j]);
      row.sort(function (a, b) { return a[0] - b[0]; });
      var kk = Math.min(k, row.length);
      kd.push(row[kk - 1][0]);
      var nn = [];
      for (j = 0; j < kk; j++) nn.push(row[j][1]);
      nb.push(nn);
    }
    /* local reachability density */
    var lrd = [];
    for (i = 0; i < n; i++) {
      var s = 0;
      for (j = 0; j < nb[i].length; j++) {
        var o = nb[i][j];
        s += Math.max(kd[o], D[i][o]);          /* reachability distance */
      }
      lrd.push(nb[i].length / Math.max(1e-12, s));
    }
    var out = [];
    for (i = 0; i < n; i++) {
      var t = 0;
      for (j = 0; j < nb[i].length; j++) t += lrd[nb[i][j]];
      out.push((t / nb[i].length) / Math.max(1e-12, lrd[i]));
    }
    return out;
  }
  /* Gaussian kernel density in 2-D, returned as a closure */
  function kde2(X, bw) {
    var n = X.length, h2 = 2 * bw * bw, c = 1 / (n * 2 * Math.PI * bw * bw);
    return function (x, y) {
      var s = 0, i, dx, dy;
      for (i = 0; i < n; i++) {
        dx = x - X[i][0]; dy = y - X[i][1];
        s += Math.exp(-(dx * dx + dy * dy) / h2);
      }
      return c * s;
    };
  }

  /* =================================================================
     4.  Isolation Forest, Liu, Ting & Zhou 2008
     ================================================================= */
  function cFactor(n) {                       /* expected path length of an unsuccessful BST search */
    if (n <= 1) return 0;
    if (n === 2) return 1;
    return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
  }
  function iTree(X, idx, depth, limit, rng, oblique) {
    if (depth >= limit || idx.length <= 1) return { leaf: true, n: idx.length };
    var p = X[0].length, i, q, lo, hi;
    if (oblique) {
      /* extended isolation forest: a random direction rather than an axis */
      var w = [], nrm = 0;
      for (i = 0; i < p; i++) { w.push(rng.n()); nrm += w[i] * w[i]; }
      nrm = Math.sqrt(nrm) || 1;
      for (i = 0; i < p; i++) w[i] /= nrm;
      var proj = [];
      for (i = 0; i < idx.length; i++) {
        var s = 0; for (q = 0; q < p; q++) s += w[q] * X[idx[i]][q];
        proj.push(s);
      }
      lo = Math.min.apply(null, proj); hi = Math.max.apply(null, proj);
      if (hi - lo < 1e-12) return { leaf: true, n: idx.length };
      var t = lo + rng.u() * (hi - lo), L = [], R = [];
      for (i = 0; i < idx.length; i++) (proj[i] < t ? L : R).push(idx[i]);
      if (!L.length || !R.length) return { leaf: true, n: idx.length };
      return { leaf: false, w: w, t: t,
               l: iTree(X, L, depth + 1, limit, rng, true), r: iTree(X, R, depth + 1, limit, rng, true) };
    }
    /* classical: a random axis and a random split inside its observed range */
    q = rng.int(p);
    lo = Infinity; hi = -Infinity;
    for (i = 0; i < idx.length; i++) { var v = X[idx[i]][q]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-12) return { leaf: true, n: idx.length };
    var th = lo + rng.u() * (hi - lo), A = [], B = [];
    for (i = 0; i < idx.length; i++) (X[idx[i]][q] < th ? A : B).push(idx[i]);
    if (!A.length || !B.length) return { leaf: true, n: idx.length };
    return { leaf: false, q: q, t: th,
             l: iTree(X, A, depth + 1, limit, rng, false), r: iTree(X, B, depth + 1, limit, rng, false) };
  }
  function pathLen(node, pt, depth) {
    if (node.leaf) return depth + cFactor(node.n);
    if (node.w) {
      var s = 0; for (var i = 0; i < node.w.length; i++) s += node.w[i] * pt[i];
      return pathLen(s < node.t ? node.l : node.r, pt, depth + 1);
    }
    return pathLen(pt[node.q] < node.t ? node.l : node.r, pt, depth + 1);
  }
  function iforest(X, opt) {
    opt = opt || {};
    var nT = opt.trees || 100, sub = Math.min(opt.sub || 256, X.length);
    var rng = new RNG(opt.seed || 11), limit = Math.ceil(Math.log(sub) / Math.LN2);
    var trees = [], t, i;
    for (t = 0; t < nT; t++) {
      var idx = [], seen = {};
      while (idx.length < sub) { var j = rng.int(X.length); if (!seen[j]) { seen[j] = 1; idx.push(j); } }
      trees.push(iTree(X, idx, 0, limit, rng, !!opt.oblique));
    }
    var cN = cFactor(sub);
    return {
      trees: trees,
      /* the published score: 2^(-E[h]/c(n)), so 0.5 is the ambiguous middle
         and values above ~0.6 are candidates */
      score: function (pt) {
        var s = 0;
        for (i = 0; i < trees.length; i++) s += pathLen(trees[i], pt, 0);
        return Math.pow(2, -(s / trees.length) / cN);
      },
      depth: function (pt) {
        var s = 0;
        for (i = 0; i < trees.length; i++) s += pathLen(trees[i], pt, 0);
        return s / trees.length;
      }
    };
  }

  /* =================================================================
     5.  PCA, and a genuinely trained tiny autoencoder
     ================================================================= */
  function pca2(X) {
    var mc = meanCov(X), e = eig2(mc.S);
    return { mu: mc.mu, S: mc.S, dir: e.v, l: e.l };
  }
  /* reconstruction error keeping one principal component */
  function pcaRecon(X, p) {
    var o = [], i;
    for (i = 0; i < X.length; i++) {
      var dx = X[i][0] - p.mu[0], dy = X[i][1] - p.mu[1];
      var t = dx * p.dir[0] + dy * p.dir[1];
      var rx = p.mu[0] + t * p.dir[0], ry = p.mu[1] + t * p.dir[1];
      o.push({ x: rx, y: ry, e: Math.hypot(X[i][0] - rx, X[i][1] - ry) });
    }
    return o;
  }

  /* A 2 -> H -> K -> H -> 2 autoencoder with tanh hidden layers, trained by
     full-batch gradient descent with momentum. Small enough to train live in
     the browser, and structured so that the failure that matters is visible:
     at K = 1 the code is a curve through the data and anomalies reconstruct
     badly; at K = 2 the code has the same dimension as the input, the network
     can learn the identity, and it reconstructs the anomalies just as well as
     the normal points. Reconstruction error then scores nothing. */
  function AE(H, K, seed) {
    K = K || 1;
    var r = new RNG(seed || 3), i, j, k;
    function mat(a, b) { var M = [], p, q; for (p = 0; p < a; p++) { M.push([]); for (q = 0; q < b; q++) M[p].push(r.n() * Math.sqrt(2 / (a + b))); } return M; }
    function zmat(a, b) { var M = [], p, q; for (p = 0; p < a; p++) { M.push([]); for (q = 0; q < b; q++) M[p].push(0); } return M; }
    function vec(a) { var v = [], p; for (p = 0; p < a; p++) v.push(0); return v; }
    var W1 = mat(2, H), b1 = vec(H), W2 = mat(H, K), b2 = vec(K),
        W3 = mat(K, H), b3 = vec(H), W4 = mat(H, 2), b4 = vec(2);
    var mW1 = zmat(2, H), mb1 = vec(H), mW2 = zmat(H, K), mb2 = vec(K),
        mW3 = zmat(K, H), mb3 = vec(H), mW4 = zmat(H, 2), mb4 = vec(2);

    function fwd(x) {
      var h1 = [], z, a, c;
      for (j = 0; j < H; j++) { z = b1[j] + x[0] * W1[0][j] + x[1] * W1[1][j]; h1.push(Math.tanh(z)); }
      var code = [];
      for (c = 0; c < K; c++) { z = b2[c]; for (j = 0; j < H; j++) z += h1[j] * W2[j][c]; code.push(z); }
      var h2 = [];
      for (j = 0; j < H; j++) { a = b3[j]; for (c = 0; c < K; c++) a += code[c] * W3[c][j]; h2.push(Math.tanh(a)); }
      var out = [b4[0], b4[1]];
      for (j = 0; j < H; j++) { out[0] += h2[j] * W4[j][0]; out[1] += h2[j] * W4[j][1]; }
      return { h1: h1, code: code, h2: h2, out: out };
    }
    this.H = H; this.K = K;
    this.forward = function (x) { return fwd(x).out; };
    this.code = function (x) { return fwd(x).code; };
    /* the decoder alone, so the learned manifold can be drawn by sweeping the code */
    this.decode = function (code) {
      var h2 = [], a, c, p;
      for (p = 0; p < H; p++) {
        a = b3[p];
        for (c = 0; c < K; c++) a += code[c] * W3[c][p];
        h2.push(Math.tanh(a));
      }
      var out = [b4[0], b4[1]];
      for (p = 0; p < H; p++) { out[0] += h2[p] * W4[p][0]; out[1] += h2[p] * W4[p][1]; }
      return out;
    };
    this.err = function (x) { var o = fwd(x).out; return Math.hypot(o[0] - x[0], o[1] - x[1]); };
    /* one pass of full-batch gradient descent with momentum */
    this.step = function (data, lr, mom) {
      var n = data.length, loss = 0, c;
      var gW1 = zmat(2, H), gb1 = vec(H), gW2 = zmat(H, K), gb2 = vec(K),
          gW3 = zmat(K, H), gb3 = vec(H), gW4 = zmat(H, 2), gb4 = vec(2);
      for (k = 0; k < n; k++) {
        var x = data[k], f = fwd(x);
        var e0 = f.out[0] - x[0], e1 = f.out[1] - x[1];
        loss += (e0 * e0 + e1 * e1) / n;
        var dO = [2 * e0 / n, 2 * e1 / n];
        gb4[0] += dO[0]; gb4[1] += dO[1];
        var dH2 = [];
        for (j = 0; j < H; j++) {
          gW4[j][0] += f.h2[j] * dO[0]; gW4[j][1] += f.h2[j] * dO[1];
          dH2.push((W4[j][0] * dO[0] + W4[j][1] * dO[1]) * (1 - f.h2[j] * f.h2[j]));
        }
        var dCode = vec(K);
        for (j = 0; j < H; j++) {
          gb3[j] += dH2[j];
          for (c = 0; c < K; c++) { gW3[c][j] += f.code[c] * dH2[j]; dCode[c] += W3[c][j] * dH2[j]; }
        }
        for (c = 0; c < K; c++) gb2[c] += dCode[c];
        var dH1 = vec(H);
        for (j = 0; j < H; j++) {
          var s = 0;
          for (c = 0; c < K; c++) { gW2[j][c] += f.h1[j] * dCode[c]; s += W2[j][c] * dCode[c]; }
          dH1[j] = s * (1 - f.h1[j] * f.h1[j]);
        }
        for (j = 0; j < H; j++) { gb1[j] += dH1[j]; gW1[0][j] += x[0] * dH1[j]; gW1[1][j] += x[1] * dH1[j]; }
      }
      function upd(W, G, M, a, b) {
        for (var p = 0; p < a; p++) for (var q = 0; q < b; q++) {
          M[p][q] = mom * M[p][q] - lr * G[p][q]; W[p][q] += M[p][q];
        }
      }
      function updv(v, g, m, a) { for (var p = 0; p < a; p++) { m[p] = mom * m[p] - lr * g[p]; v[p] += m[p]; } }
      upd(W1, gW1, mW1, 2, H); updv(b1, gb1, mb1, H);
      upd(W2, gW2, mW2, H, K); updv(b2, gb2, mb2, K);
      upd(W3, gW3, mW3, K, H); updv(b3, gb3, mb3, H);
      upd(W4, gW4, mW4, H, 2); updv(b4, gb4, mb4, 2);
      return loss;
    };
  }

  /* =================================================================
     6.  time series
     ================================================================= */
  function movmed(x, k) {
    var o = [], i, j, w;
    for (i = 0; i < x.length; i++) {
      w = [];
      for (j = Math.max(0, i - k); j <= Math.min(x.length - 1, i + k); j++) w.push(x[j]);
      o.push(median(w));
    }
    return o;
  }
  /* A deliberately simple seasonal-trend decomposition: robust trend by
     running median, seasonal by the median of each phase of the detrended
     series, remainder by subtraction. Not STL, but the same three parts and
     the same lesson. */
  function decompose(x, period) {
    var trend = movmed(x, Math.max(2, Math.round(period * 0.75))), i;
    var det = [];
    for (i = 0; i < x.length; i++) det.push(x[i] - trend[i]);
    var buckets = [], p;
    for (p = 0; p < period; p++) buckets.push([]);
    for (i = 0; i < det.length; i++) buckets[i % period].push(det[i]);
    var seas0 = [];
    for (p = 0; p < period; p++) seas0.push(median(buckets[p]));
    var m = mean(seas0);
    for (p = 0; p < period; p++) seas0[p] -= m;
    var seasonal = [], resid = [];
    for (i = 0; i < x.length; i++) {
      seasonal.push(seas0[i % period]);
      resid.push(x[i] - trend[i] - seas0[i % period]);
    }
    return { trend: trend, seasonal: seasonal, resid: resid };
  }
  /* CUSUM, two-sided, on a standardised series */
  function cusum(x, k) {
    var m = median(x), s = 1.4826 * (mad(x) || 1e-9), i, hi = 0, lo = 0, o = [];
    k = (k === undefined ? 0.5 : k);
    for (i = 0; i < x.length; i++) {
      var z = (x[i] - m) / s;
      hi = Math.max(0, hi + z - k);
      lo = Math.max(0, lo - z - k);
      o.push(Math.max(hi, lo));
    }
    return o;
  }
  /* matrix profile by brute force: for each window, the z-normalised
     Euclidean distance to its nearest non-trivial match. The largest values
     are the discords - the collective anomalies. */
  function matrixProfile(x, m) {
    var n = x.length - m + 1, i, j, k, prof = [], znorm = [];
    for (i = 0; i < n; i++) {
      var w = x.slice(i, i + m), mu = mean(w), s = sd(w) || 1e-9, z = new Float64Array(m);
      for (k = 0; k < m; k++) z[k] = (w[k] - mu) / s;
      znorm.push(z);
    }
    var excl = Math.ceil(m / 2);
    for (i = 0; i < n; i++) {
      var best = Infinity;
      for (j = 0; j < n; j++) {
        if (Math.abs(i - j) < excl) continue;
        var d = 0, a = znorm[i], b = znorm[j];
        for (k = 0; k < m; k++) { var t = a[k] - b[k]; d += t * t; if (d >= best) break; }
        if (d < best) best = d;
      }
      prof.push(Math.sqrt(best));
    }
    return prof;
  }

  /* =================================================================
     7.  evaluation
     ================================================================= */
  function rank(scores) {                    /* descending order of index */
    var o = [], i;
    for (i = 0; i < scores.length; i++) o.push(i);
    o.sort(function (a, b) { return scores[b] - scores[a]; });
    return o;
  }
  function roc(scores, y) {
    var ord = rank(scores), P = 0, N = 0, i;
    for (i = 0; i < y.length; i++) y[i] ? P++ : N++;
    var tp = 0, fp = 0, fpr = [0], tpr = [0], auc = 0, prevF = 0, prevT = 0;
    for (i = 0; i < ord.length; i++) {
      y[ord[i]] ? tp++ : fp++;
      var F = fp / Math.max(1, N), T = tp / Math.max(1, P);
      fpr.push(F); tpr.push(T);
      auc += (F - prevF) * (T + prevT) / 2;
      prevF = F; prevT = T;
    }
    return { fpr: fpr, tpr: tpr, auc: auc };
  }
  function pr(scores, y) {
    var ord = rank(scores), P = 0, i;
    for (i = 0; i < y.length; i++) if (y[i]) P++;
    var tp = 0, fp = 0, rec = [], prec = [], ap = 0, prevR = 0;
    for (i = 0; i < ord.length; i++) {
      y[ord[i]] ? tp++ : fp++;
      var R = tp / Math.max(1, P), Q = tp / (tp + fp);
      rec.push(R); prec.push(Q);
      ap += (R - prevR) * Q;                 /* average precision, the step-wise form */
      prevR = R;
    }
    return { rec: rec, prec: prec, ap: ap, base: P / y.length };
  }
  function precAtK(scores, y, k) {
    var ord = rank(scores), t = 0, i;
    for (i = 0; i < Math.min(k, ord.length); i++) if (y[ord[i]]) t++;
    return t / Math.max(1, Math.min(k, ord.length));
  }
  /* The point-adjustment protocol: if any point inside a true anomalous
     segment is flagged, the whole segment counts as detected. This is what
     inflates published time-series F1 scores. */
  function pointAdjust(pred, y) {
    var out = pred.slice(), i, j, s;
    i = 0;
    while (i < y.length) {
      if (!y[i]) { i++; continue; }
      s = i;
      while (i < y.length && y[i]) i++;
      var any = false;
      for (j = s; j < i; j++) if (pred[j]) any = true;
      if (any) for (j = s; j < i; j++) out[j] = 1;
    }
    return out;
  }
  function f1(pred, y) {
    var tp = 0, fp = 0, fn = 0, i;
    for (i = 0; i < y.length; i++) {
      if (pred[i] && y[i]) tp++;
      else if (pred[i] && !y[i]) fp++;
      else if (!pred[i] && y[i]) fn++;
    }
    var p = tp / Math.max(1, tp + fp), r = tp / Math.max(1, tp + fn);
    return { p: p, r: r, f1: 2 * p * r / Math.max(1e-12, p + r) };
  }

  /* =================================================================
     8.  synthetic data the demonstrations share
     ================================================================= */
  /* two Gaussian clusters of deliberately different density, plus a few
     genuine outliers: the configuration LOF was invented for */
  function twoDensity(n, seed) {
    var r = new RNG(seed || 5), X = [], y = [], i;
    var nA = Math.round(n * 0.55), nB = n - nA - 6;
    for (i = 0; i < nA; i++) { X.push([-1.5 + 0.30 * r.n(), 0.2 + 0.30 * r.n()]); y.push(0); }
    for (i = 0; i < nB; i++) { X.push([1.7 + 0.95 * r.n(), -0.1 + 0.95 * r.n()]); y.push(0); }
    var out = [[-0.15, 1.15], [-2.6, 1.55], [0.15, -1.55], [-1.55, -1.35], [3.7, 1.9], [-3.1, -0.4]];
    for (i = 0; i < out.length; i++) { X.push(out[i]); y.push(1); }
    return { X: X, y: y };
  }
  /* a clean Gaussian sample with a controllable fraction of gross outliers,
     all at the same place. This is the masking demonstration: as the fraction
     grows the outliers inflate the standard deviation enough to hide
     themselves from the rule that is meant to find them. */
  function masking(n, cfrac, at, seed) {
    var r = new RNG(seed || 31), x = [], y = [], i, nc = Math.round(n * cfrac);
    for (i = 0; i < n - nc; i++) { x.push(r.n()); y.push(0); }
    for (i = 0; i < nc; i++) { x.push((at === undefined ? 5 : at) + 0.28 * r.n()); y.push(1); }
    return { x: x, y: y };
  }
  /* normal data on a diagonal band; the anomalies sit off the band but well
     inside the observed range of each coordinate separately. Nothing is
     extreme in x, nothing is extreme in y, and only the combination is odd -
     which is exactly what axis-parallel splits cannot see. */
  function diagonalBand(n, seed) {
    var r = new RNG(seed || 41), X = [], y = [], i, t;
    for (i = 0; i < n; i++) {
      t = -2.2 + 4.4 * r.u();
      X.push([t + 0.16 * r.n(), t + 0.16 * r.n()]); y.push(0);
    }
    var an = [[-1.5, 1.4], [1.5, -1.4], [-0.9, 1.0], [1.0, -0.9], [0.4, -1.7], [-1.7, 0.4]];
    for (i = 0; i < an.length; i++) { X.push(an[i]); y.push(1); }
    return { X: X, y: y };
  }
  /* an elongated correlated cloud with a contaminating clump: the case that
     breaks the classical covariance estimate */
  function correlatedWithClump(n, cfrac, seed) {
    var r = new RNG(seed || 9), X = [], y = [], i, nc = Math.round(n * cfrac);
    for (i = 0; i < n - nc; i++) {
      var a = 1.25 * r.n(), b = 0.32 * r.n();
      X.push([a, 0.78 * a + b]); y.push(0);
    }
    for (i = 0; i < nc; i++) { X.push([1.9 + 0.22 * r.n(), -1.5 + 0.22 * r.n()]); y.push(1); }
    return { X: X, y: y };
  }
  /* a curved one-dimensional manifold with off-manifold anomalies: linear PCA
     cannot follow it, an autoencoder can */
  function arcManifold(n, seed) {
    var r = new RNG(seed || 13), X = [], y = [], i, t;
    for (i = 0; i < n; i++) {
      t = -1.6 + 3.2 * (i / (n - 1)) + 0.02 * r.n();
      X.push([t + 0.055 * r.n(), 0.85 * t * t - 0.8 + 0.055 * r.n()]); y.push(0);
    }
    var an = [[-1.35, 0.95], [0.05, 1.45], [1.25, -0.55], [-0.55, -1.15], [1.45, 1.75], [-1.05, -0.85]];
    for (i = 0; i < an.length; i++) { X.push(an[i]); y.push(1); }
    return { X: X, y: y };
  }
  /* a seasonal series carrying one of each kind of anomaly */
  function seasonalSeries(n, seed) {
    var r = new RNG(seed || 17), P = 48, x = [], y = [], i;
    for (i = 0; i < n; i++) {
      var trend = 0.0016 * i;
      var seas = 1.0 * Math.sin(2 * Math.PI * i / P) + 0.34 * Math.sin(4 * Math.PI * i / P + 0.7);
      x.push(trend + seas + 0.085 * r.n()); y.push(0);
    }
    /* (i) point anomaly: a single spike, obvious in the raw series */
    var ip = 120; x[ip] += 3.1; y[ip] = 1;
    /* (ii) contextual anomaly: a value well inside the global range, but at
       the wrong phase of the cycle */
    var ic = 268;
    for (i = ic; i < ic + 3; i++) { x[i] = 1.25 + 0.05 * r.n(); y[i] = 1; }
    /* (iii) collective anomaly: the oscillation flattens for a whole cycle.
       Every reading stays in range; the sequence is wrong. */
    var i0 = 400, i1 = 446;
    for (i = i0; i < i1; i++) { x[i] = 0.0016 * i + 0.26 * Math.sin(2 * Math.PI * i / 11) + 0.085 * r.n(); y[i] = 1; }
    return { x: x, y: y, period: P, marks: { point: ip, ctx: [ic, ic + 3], coll: [i0, i1] } };
  }
  /* scores from a detector of given separation, at a given contamination */
  function scoreSet(n, contam, sep, seed) {
    var r = new RNG(seed || 23), s = [], y = [], i, nA = Math.max(1, Math.round(n * contam));
    for (i = 0; i < n - nA; i++) { s.push(r.n()); y.push(0); }
    for (i = 0; i < nA; i++) { s.push(sep + r.n()); y.push(1); }
    return { s: s, y: y };
  }

  /* =================================================================
     9.  small shared utilities for the drawing code
     ================================================================= */
  function minmax(a) {
    var lo = Infinity, hi = -Infinity, i;
    for (i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
    return [lo, hi];
  }
  /* draw an image-data field without falling foul of the canvas transform:
     stage it on an offscreen canvas, then drawImage */
  function blit(ctx, img, x, y, w, h) {
    var off = document.createElement('canvas');
    off.width = img.width; off.height = img.height;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.drawImage(off, Math.floor(x), Math.floor(y), w, h);
  }
  /* marching squares on a scalar field, for iso-contours */
  function contour(F, gw, gh, level, x0, y0, sx, sy) {
    var segs = [], i, j;
    function ip(a, b, va, vb) { var t = (level - va) / ((vb - va) || 1e-12); return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]; }
    for (j = 0; j < gh - 1; j++) for (i = 0; i < gw - 1; i++) {
      var v00 = F[j * gw + i], v10 = F[j * gw + i + 1], v01 = F[(j + 1) * gw + i], v11 = F[(j + 1) * gw + i + 1];
      var p00 = [x0 + i * sx, y0 + j * sy], p10 = [x0 + (i + 1) * sx, y0 + j * sy],
          p01 = [x0 + i * sx, y0 + (j + 1) * sy], p11 = [x0 + (i + 1) * sx, y0 + (j + 1) * sy];
      var c = (v00 > level ? 1 : 0) | (v10 > level ? 2 : 0) | (v11 > level ? 4 : 0) | (v01 > level ? 8 : 0);
      var T = ip(p00, p10, v00, v10), R = ip(p10, p11, v10, v11),
          B = ip(p01, p11, v01, v11), L = ip(p00, p01, v00, v01);
      switch (c) {
        case 1: case 14: segs.push([L, T]); break;
        case 2: case 13: segs.push([T, R]); break;
        case 3: case 12: segs.push([L, R]); break;
        case 4: case 11: segs.push([R, B]); break;
        case 5: segs.push([L, T]); segs.push([R, B]); break;
        case 6: case 9: segs.push([T, B]); break;
        case 7: case 8: segs.push([L, B]); break;
        case 10: segs.push([L, B]); segs.push([T, R]); break;
      }
    }
    return segs;
  }
  /* the ColorBrewer Blues ramp, light to dark */
  var BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
  function blues(t) {
    if (!isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));
    var i = Math.min(BLUES.length - 1, Math.floor(t * (BLUES.length - 1e-9)));
    return BLUES[i];
  }
  function bluesRGB(t) {
    var h = blues(t);
    return [parseInt(h.substr(1, 2), 16), parseInt(h.substr(3, 2), 16), parseInt(h.substr(5, 2), 16)];
  }

  return {
    RNG: RNG,
    mean: mean, sd: sd, median: median, quantile: quantile, mad: mad,
    zscore: zscore, modz: modz, tukey: tukey,
    meanCov: meanCov, inv2: inv2, mahal2: mahal2, mahalPoint: mahalPoint,
    mcd2: mcd2, eig2: eig2, ellipse: ellipse,
    distMatrix: distMatrix, knnDist: knnDist, lof: lof, kde2: kde2,
    iforest: iforest, cFactor: cFactor,
    pca2: pca2, pcaRecon: pcaRecon, AE: AE,
    movmed: movmed, decompose: decompose, cusum: cusum, matrixProfile: matrixProfile,
    roc: roc, pr: pr, precAtK: precAtK, pointAdjust: pointAdjust, f1: f1, rank: rank,
    twoDensity: twoDensity, correlatedWithClump: correlatedWithClump,
    masking: masking, diagonalBand: diagonalBand,
    arcManifold: arcManifold, seasonalSeries: seasonalSeries, scoreSet: scoreSet,
    minmax: minmax, blit: blit, contour: contour, blues: blues, bluesRGB: bluesRGB, BLUES: BLUES
  };
})();
