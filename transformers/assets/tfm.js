/* ---------------------------------------------------------------------------
   A transformer small enough to train in a browser tab.

   Reverse-mode automatic differentiation over dense matrices, plus the three
   operations a transformer needs that an MLP does not: A B^T for the attention
   scores, a row-wise softmax, and a second matrix product against the values.
   Everything else - residual connections, a feed-forward block, a linear
   readout - falls out of the same tape.
   --------------------------------------------------------------------------- */
var TFM = (function () {

  /* =======================================================================
     Tape
     ======================================================================= */
  var TAPE = [];
  function nd(v, n, m, bw) {
    var t = {v: v, g: new Float64Array(v.length), n: n, m: m, bw: bw};
    TAPE.push(t); return t;
  }
  function leaf(v, n, m) { return {v: v, g: new Float64Array(v.length), n: n, m: m, bw: null}; }
  function reset() { TAPE.length = 0; }
  function backward(loss) {
    loss.g[0] = 1;
    for (var i = TAPE.length - 1; i >= 0; i--) if (TAPE[i].bw) TAPE[i].bw(TAPE[i]);
  }

  /* ---------------- basic operations ---------------- */

  /* A [n x p] * B [p x q] -> [n x q].  B may be a parameter or an activation;
     either way its gradient is accumulated, which is what lets the same routine
     serve both the weight matrices and the attention-times-values product. */
  function mm(A, B) {
    var n = A.n, p = A.m, q = B.m, out = new Float64Array(n * q), i, k, j, a;
    for (i = 0; i < n; i++)
      for (k = 0; k < p; k++) {
        a = A.v[i * p + k]; if (a === 0) continue;
        for (j = 0; j < q; j++) out[i * q + j] += a * B.v[k * q + j];
      }
    return nd(out, n, q, function (t) {
      var i, k, j, s, g, av;
      for (i = 0; i < n; i++)
        for (k = 0; k < p; k++) {
          s = 0; av = A.v[i * p + k];
          for (j = 0; j < q; j++) {
            g = t.g[i * q + j];
            s += g * B.v[k * q + j];
            B.g[k * q + j] += av * g;
          }
          A.g[i * p + k] += s;
        }
    });
  }

  /* A [n x d] * B^T where B is [m x d]  ->  [n x m].  The attention scores. */
  function mmNT(A, B) {
    var n = A.n, d = A.m, m = B.n, out = new Float64Array(n * m), i, j, k, s;
    for (i = 0; i < n; i++)
      for (j = 0; j < m; j++) {
        s = 0;
        for (k = 0; k < d; k++) s += A.v[i * d + k] * B.v[j * d + k];
        out[i * m + j] = s;
      }
    return nd(out, n, m, function (t) {
      var i, j, k, g;
      for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
          g = t.g[i * m + j]; if (g === 0) continue;
          for (k = 0; k < d; k++) {
            A.g[i * d + k] += g * B.v[j * d + k];
            B.g[j * d + k] += g * A.v[i * d + k];
          }
        }
    });
  }

  /* Row-wise softmax.  The Jacobian of a softmax row is
     diag(p) - p p^T, so the backward pass is  dx = p * (dy - <dy, p>). */
  function softmaxRows(A, mask) {
    var n = A.n, m = A.m, out = new Float64Array(n * m), i, j, mx, s;
    for (i = 0; i < n; i++) {
      mx = -Infinity;
      for (j = 0; j < m; j++) {
        if (mask && !mask(i, j)) continue;
        if (A.v[i * m + j] > mx) mx = A.v[i * m + j];
      }
      s = 0;
      for (j = 0; j < m; j++) {
        if (mask && !mask(i, j)) { out[i * m + j] = 0; continue; }
        out[i * m + j] = Math.exp(A.v[i * m + j] - mx); s += out[i * m + j];
      }
      if (s > 0) for (j = 0; j < m; j++) out[i * m + j] /= s;
    }
    return nd(out, n, m, function (t) {
      var i, j, dot;
      for (i = 0; i < n; i++) {
        dot = 0;
        for (j = 0; j < m; j++) dot += t.g[i * m + j] * out[i * m + j];
        for (j = 0; j < m; j++)
          A.g[i * m + j] += out[i * m + j] * (t.g[i * m + j] - dot);
      }
    });
  }

  function addb(A, b) {
    var n = A.n, q = A.m, out = new Float64Array(n * q), i, j;
    for (i = 0; i < n; i++) for (j = 0; j < q; j++) out[i * q + j] = A.v[i * q + j] + b.v[j];
    return nd(out, n, q, function (t) {
      for (var i = 0; i < n; i++) for (var j = 0; j < q; j++) {
        A.g[i * q + j] += t.g[i * q + j]; b.g[j] += t.g[i * q + j];
      }
    });
  }

  function add(A, B) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = A.v[i] + B.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) { A.g[i] += t.g[i]; B.g[i] += t.g[i]; }
    });
  }

  function scale(a, A) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = a * A.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += a * t.g[i];
    });
  }

  function tanhx(A) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = Math.tanh(A.v[i]);
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += t.g[i] * (1 - out[i] * out[i]);
    });
  }

  /* Layer normalisation, per row.  Included because without it the residual
     stream drifts and training is visibly worse. */
  function layernorm(A, g, b) {
    var n = A.n, m = A.m, out = new Float64Array(n * m);
    var mu = new Float64Array(n), iv = new Float64Array(n), i, j, s, v, eps = 1e-5;
    for (i = 0; i < n; i++) {
      s = 0; for (j = 0; j < m; j++) s += A.v[i * m + j]; mu[i] = s / m;
      v = 0; for (j = 0; j < m; j++) { var d = A.v[i * m + j] - mu[i]; v += d * d; }
      iv[i] = 1 / Math.sqrt(v / m + eps);
      for (j = 0; j < m; j++) out[i * m + j] = (A.v[i * m + j] - mu[i]) * iv[i] * g.v[j] + b.v[j];
    }
    return nd(out, n, m, function (t) {
      var i, j, xh, sg = new Float64Array(m);
      for (i = 0; i < n; i++) {
        var s1 = 0, s2 = 0;
        for (j = 0; j < m; j++) {
          xh = (A.v[i * m + j] - mu[i]) * iv[i];
          var dy = t.g[i * m + j];
          g.g[j] += dy * xh; b.g[j] += dy;
          var dxh = dy * g.v[j];
          s1 += dxh; s2 += dxh * xh;
        }
        for (j = 0; j < m; j++) {
          xh = (A.v[i * m + j] - mu[i]) * iv[i];
          A.g[i * m + j] += iv[i] / m * (m * t.g[i * m + j] * g.v[j] - s1 - xh * s2);
        }
      }
    });
  }

  /* mean of (A - target)^2 over all entries, as a scalar node */
  function mse(A, target) {
    var n = A.v.length, s = 0, i, d;
    for (i = 0; i < n; i++) { d = A.v[i] - target[i]; s += d * d; }
    var out = new Float64Array(1); out[0] = s / n;
    return nd(out, 1, 1, function (t) {
      for (var i = 0; i < n; i++) A.g[i] += t.g[0] * 2 * (A.v[i] - target[i]) / n;
    });
  }

  /* Mean logistic loss over a column of logits.  Z is [n x 1]; y[i] is 0 or 1.
     loss = mean( log(1+e^z) - y z ),   d loss / dz = ( sigmoid(z) - y ) / n.
     The log1p form avoids overflow for large |z|. */
  function bce(Z, y) {
    var n = Z.v.length, s = 0, i, z;
    for (i = 0; i < n; i++) {
      z = Z.v[i];
      s += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - y[i] * z;
    }
    var out = new Float64Array(1); out[0] = s / n;
    return nd(out, 1, 1, function (t) {
      for (var i = 0; i < n; i++) {
        var p = 1 / (1 + Math.exp(-Z.v[i]));
        Z.g[i] += t.g[0] * (p - y[i]) / n;
      }
    });
  }

  /* scalar addition, on the tape.  Needed to sum the per-sample losses of a
     minibatch into a single node: a sum built outside the tape is invisible to
     the reverse sweep, and the whole batch silently contributes no gradient. */
  function addS(a, b) {
    var out = new Float64Array(1); out[0] = a.v[0] + b.v[0];
    return nd(out, 1, 1, function (t) { a.g[0] += t.g[0]; b.g[0] += t.g[0]; });
  }
  function sumS(list) {
    var s = list[0];
    for (var i = 1; i < list.length; i++) s = addS(s, list[i]);
    return s;
  }

  /* pick one row of a matrix as a [1 x m] node */
  function row(A, r) {
    var m = A.m, out = new Float64Array(m), j;
    for (j = 0; j < m; j++) out[j] = A.v[r * m + j];
    return nd(out, 1, m, function (t) {
      for (var j = 0; j < m; j++) A.g[r * m + j] += t.g[j];
    });
  }

  /* =======================================================================
     Random numbers
     ======================================================================= */
  function Rng(seed) {
    var s = (seed >>> 0) || 99;
    this.u = function () { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
    var sp = null;
    this.n = function () {
      if (sp !== null) { var v = sp; sp = null; return v; }
      var a = this.u() || 1e-12, b = this.u(), r = Math.sqrt(-2 * Math.log(a));
      sp = r * Math.sin(2 * Math.PI * b); return r * Math.cos(2 * Math.PI * b);
    };
  }

  /* =======================================================================
     The model: embed -> [attention + feed-forward] -> read out the last row
     ======================================================================= */
  function Model(opt) {
    opt = opt || {};
    var din = opt.din || 6, dm = opt.dmodel || 16, dk = opt.dk || 8, dff = opt.dff || 24;
    var rng = new Rng(opt.seed || 3), i;
    this.din = din; this.dm = dm; this.dk = dk;
    /* switches, so that the architecture page can ablate one piece at a time.
       noscale drops the 1/sqrt(d_k); nores removes the two residual additions;
       nonorm makes both layer norms the identity. */
    this.noscale = !!opt.noscale; this.nores = !!opt.nores; this.nonorm = !!opt.nonorm;
    this.causal = !!opt.causal;
    function M(a, b, s) {
      var W = new Float64Array(a * b);
      for (var k = 0; k < W.length; k++) W[k] = rng.n() * s;
      return leaf(W, a, b);
    }
    function V(b, val) {
      var x = new Float64Array(b);
      if (val !== undefined) for (var k = 0; k < b; k++) x[k] = val;
      return leaf(x, 1, b);
    }
    this.Wemb = M(din, dm, Math.sqrt(1 / din));  this.bemb = V(dm);
    this.Wq = M(dm, dk, Math.sqrt(1 / dm));
    this.Wk = M(dm, dk, Math.sqrt(1 / dm));
    this.Wv = M(dm, dm, Math.sqrt(1 / dm));
    this.g1 = V(dm, 1); this.b1 = V(dm);
    this.g2 = V(dm, 1); this.b2 = V(dm);
    this.W1 = M(dm, dff, Math.sqrt(1 / dm)); this.bf1 = V(dff);
    this.W2 = M(dff, dm, Math.sqrt(1 / dff)); this.bf2 = V(dm);
    this.Wout = M(dm, 1, Math.sqrt(1 / dm)); this.bout = V(1);
    this.lastAttn = null;
  }
  Model.prototype.params = function () {
    return [this.Wemb, this.bemb, this.Wq, this.Wk, this.Wv, this.g1, this.b1,
            this.g2, this.b2, this.W1, this.bf1, this.W2, this.bf2, this.Wout, this.bout];
  };
  Model.prototype.count = function () {
    var n = 0, p = this.params();
    for (var i = 0; i < p.length; i++) n += p[i].v.length;
    return n;
  };

  /* X: Float64Array of shape [T x din].  Returns the scalar prediction from the
     final position, and stashes the attention matrix for drawing.

     With opt.causal the softmax is masked so that position i sees only
     positions j <= i, and forwardAll returns one output per position instead
     of only the last.  That is the whole difference between a model that reads
     and a model that generates. */
  Model.prototype.forwardAll = function (X, T) {
    var x = leaf(X, T, this.din);
    var h = addb(mm(x, this.Wemb), this.bemb);
    var hn = this.nonorm ? h : layernorm(h, this.g1, this.b1);
    var Q = mm(hn, this.Wq), K = mm(hn, this.Wk), Vv = mm(hn, this.Wv);
    var sc = this.noscale ? 1 : 1 / Math.sqrt(this.dk);
    var S = scale(sc, mmNT(Q, K));
    var A = softmaxRows(S, this.causal ? function (i, j) { return j <= i; } : null);
    this.lastAttn = A;
    var att = mm(A, Vv);
    h = this.nores ? att : add(h, att);
    var h2 = this.nonorm ? h : layernorm(h, this.g2, this.b2);
    var ff = mm(tanhx(addb(mm(h2, this.W1), this.bf1)), this.W2);
    h = this.nores ? ff : add(h, ff);
    h = addb(h, this.bf2);
    return addb(mm(h, this.Wout), this.bout);             /* [T x 1]        */
  };
  Model.prototype.forward = function (X, T) {
    var x = leaf(X, T, this.din);
    var h = addb(mm(x, this.Wemb), this.bemb);            /* embed          */
    var hn = this.nonorm ? h : layernorm(h, this.g1, this.b1);   /* pre-norm */
    var Q = mm(hn, this.Wq), K = mm(hn, this.Wk), Vv = mm(hn, this.Wv);
    var sc = this.noscale ? 1 : 1 / Math.sqrt(this.dk);
    var S = scale(sc, mmNT(Q, K));                        /* scores         */
    var A = softmaxRows(S);                               /* attention      */
    this.lastAttn = A;
    var att = mm(A, Vv);
    h = this.nores ? att : add(h, att);                   /* residual       */
    var h2 = this.nonorm ? h : layernorm(h, this.g2, this.b2);
    var ff = mm(tanhx(addb(mm(h2, this.W1), this.bf1)), this.W2);
    h = this.nores ? ff : add(h, ff);
    h = addb(h, this.bf2);
    var last = row(h, T - 1);
    return addb(mm(last, this.Wout), this.bout);          /* [1 x 1]        */
  };

  /* =======================================================================
     A plain MLP on the flattened input, for comparison.

     It sees exactly the same numbers as the transformer, but as one long
     vector of length T*din.  That fixes T at construction: the same network
     cannot be shown a different number of measurements, which is the point
     of the comparison on the "why physics" page.
     ======================================================================= */
  function Mlp(opt) {
    opt = opt || {};
    var T = opt.T || 6, din = opt.din || 7, dh = opt.dh || 48;
    var rng = new Rng(opt.seed || 3);
    function M(a, b, s) {
      var W = new Float64Array(a * b);
      for (var k = 0; k < W.length; k++) W[k] = rng.n() * s;
      return leaf(W, a, b);
    }
    function V(b) { return leaf(new Float64Array(b), 1, b); }
    this.T = T; this.din = din; this.nin = T * din;
    this.W1 = M(this.nin, dh, Math.sqrt(1 / this.nin)); this.b1 = V(dh);
    this.W2 = M(dh, dh, Math.sqrt(1 / dh)); this.b2 = V(dh);
    this.W3 = M(dh, 1, Math.sqrt(1 / dh)); this.b3 = V(1);
  }
  Mlp.prototype.params = function () {
    return [this.W1, this.b1, this.W2, this.b2, this.W3, this.b3];
  };
  Mlp.prototype.count = function () {
    var n = 0, p = this.params();
    for (var i = 0; i < p.length; i++) n += p[i].v.length;
    return n;
  };
  Mlp.prototype.forward = function (X) {
    var x = leaf(X, 1, this.nin);
    var h = tanhx(addb(mm(x, this.W1), this.b1));
    h = tanhx(addb(mm(h, this.W2), this.b2));
    return addb(mm(h, this.W3), this.b3);
  };

  /* =======================================================================
     Adam
     ======================================================================= */
  function Adam(params, lr) {
    this.p = params; this.lr = lr === undefined ? 3e-3 : lr; this.t = 0;
    this.m = []; this.v = [];
    for (var i = 0; i < params.length; i++) {
      this.m.push(new Float64Array(params[i].v.length));
      this.v.push(new Float64Array(params[i].v.length));
    }
  }
  Adam.prototype.zero = function () { for (var i = 0; i < this.p.length; i++) this.p[i].g.fill(0); };
  Adam.prototype.step = function () {
    this.t++;
    var b1 = 0.9, b2 = 0.999, e = 1e-8,
        c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t), i, k;
    for (i = 0; i < this.p.length; i++) {
      var p = this.p[i], m = this.m[i], v = this.v[i];
      for (k = 0; k < p.v.length; k++) {
        var g = p.g[k];
        m[k] = b1 * m[k] + (1 - b1) * g;
        v[k] = b2 * v[k] + (1 - b2) * g * g;
        p.v[k] -= this.lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + e);
      }
    }
  };

  /* =======================================================================
     The task: associative recall, dressed as a physics lookup.

     T-1 "measurement" tokens, each carrying a one-hot label and a value, then
     a final "query" token carrying one of the labels and no value.  The model
     must output the value belonging to the queried label.

     Single-head attention solves this exactly: match the query label against
     the keys, then copy the corresponding value.  Which means the attention
     row for the last token should converge to a one-hot vector pointing at the
     matching measurement -- and that is the thing worth watching.
     ======================================================================= */
  function makeTask(opt) {
    opt = opt || {};
    var T = opt.T || 6, nlab = opt.nlab || (T - 1);
    var din = nlab + 2;                                   /* label, value, isquery */
    return {
      T: T, nlab: nlab, din: din,
      sample: function (rng) {
        var X = new Float64Array(T * din), labs = [], i, j;
        for (i = 0; i < nlab; i++) labs.push(i);
        for (i = nlab - 1; i > 0; i--) {                  /* shuffle the labels */
          j = Math.floor(rng.u() * (i + 1));
          var tmp = labs[i]; labs[i] = labs[j]; labs[j] = tmp;
        }
        var vals = [];
        for (i = 0; i < T - 1; i++) {
          X[i * din + labs[i]] = 1;                       /* one-hot label   */
          var v = 2 * rng.u() - 1;
          X[i * din + nlab] = v;                          /* the measurement */
          vals.push(v);
        }
        var q = Math.floor(rng.u() * (T - 1));            /* which to ask for */
        X[(T - 1) * din + labs[q]] = 1;
        X[(T - 1) * din + nlab + 1] = 1;                  /* the query flag  */
        return {X: X, y: vals[q], target: q};
      }
    };
  }

  /* =======================================================================
     A jet-substructure toy.

     Each example is a variable-size set of constituents in an (eta, phi) plane
     carrying a transverse-momentum fraction, drawn either from one prong or
     from two.  A final "class" token, carrying no kinematics, is where the
     answer is read out.  The classification is genuinely non-local: no single
     constituent tells you the answer, only the pattern of separations does.

     Token features: [log pT fraction, deta, dphi, (deta^2+dphi^2), is-class].
     The quadratic feature is optional, and switching it off is the point of
     the demonstration: without it the same model stays near chance.
     ======================================================================= */
  function makeJets(opt) {
    opt = opt || {};
    var nmin = opt.nmin || 8, nmax = opt.nmax || 16, R = opt.R || 0.4;
    var w = opt.width || 0.04, minsep = opt.minsep || 0.40;
    var quad = opt.quad !== false, din = quad ? 5 : 4;
    return {
      din: din, quad: quad, nmin: nmin, nmax: nmax, R: R,
      sample: function (rng) {
        var two = rng.u() < 0.5 ? 1 : 0;
        var n = nmin + Math.floor(rng.u() * (nmax - nmin + 1));
        var T = n + 1, X = new Float64Array(T * din), c = [], k, th, r, i;
        for (k = 0; k < (two ? 2 : 1); k++) {
          th = 2 * Math.PI * rng.u(); r = R * Math.sqrt(rng.u());
          c.push([r * Math.cos(th), r * Math.sin(th)]);
        }
        if (two) {                                   /* keep the prongs apart */
          var dx = c[1][0] - c[0][0], dy = c[1][1] - c[0][1];
          if (Math.sqrt(dx * dx + dy * dy) < minsep) {
            var a = 2 * Math.PI * rng.u();
            c[1][0] = c[0][0] + minsep * Math.cos(a);
            c[1][1] = c[0][1] + minsep * Math.sin(a);
          }
        }
        var pts = [], ps = 0;
        for (i = 0; i < n; i++) {
          var g = two ? (rng.u() < 0.5 ? 0 : 1) : 0;
          var p = Math.exp(-2 * rng.u());
          pts.push([c[g][0] + w * rng.n(), c[g][1] + w * rng.n(), p]); ps += p;
        }
        for (i = 0; i < n; i++) {
          X[i * din + 0] = Math.log(pts[i][2] / ps) + 2.5;   /* ~ O(1)      */
          X[i * din + 1] = pts[i][0];
          X[i * din + 2] = pts[i][1];
          if (quad) X[i * din + 3] = pts[i][0] * pts[i][0] + pts[i][1] * pts[i][1];
        }
        X[n * din + din - 1] = 1;                            /* class token */
        return {X: X, T: T, n: n, y: two ? 1 : -1, pts: pts, two: two};
      }
    };
  }

  /* =======================================================================
     Masses on a ring.

     N masses joined by springs, so that the acceleration of each is a linear
     combination of all the displacements,  a = -K x.  Three couplings:
     nearest neighbour, an inverse-square falloff around the ring, and uniform
     all-to-all.  The network is shown the displacements and the position of
     each mass on the ring, and must produce the accelerations.

     The point of the exercise: the attention matrix it learns should look
     like K.  A local coupling should give a banded A; a long-range one a
     broad A.  Nobody tells it the interaction range.
     ======================================================================= */
  function makeRing(opt) {
    opt = opt || {};
    var N = opt.N || 12, kind = opt.kind || 'nn', din = 3, i, j;
    var K = [];
    for (i = 0; i < N; i++) K.push(new Float64Array(N));
    function dist(a, b) { var d = Math.abs(a - b); return Math.min(d, N - d); }
    if (kind === 'nn') {
      for (i = 0; i < N; i++) { K[i][i] = 2; K[i][(i + 1) % N] = -1; K[i][(i + N - 1) % N] = -1; }
    } else if (kind === 'long') {
      for (i = 0; i < N; i++) {
        var s = 0;
        for (j = 0; j < N; j++) if (j !== i) { var w = 1 / (dist(i, j) * dist(i, j)); K[i][j] = -w; s += w; }
        K[i][i] = s;
      }
    } else {
      for (i = 0; i < N; i++) { for (j = 0; j < N; j++) if (j !== i) K[i][j] = -1 / (N - 1); K[i][i] = 1; }
    }
    return {
      N: N, din: din, K: K, kind: kind, dist: dist,
      /* a smooth random displacement field, built from three Fourier modes */
      sample: function (rng) {
        var X = new Float64Array(N * din), x = [], amp = [], ph = [], m, i, j;
        for (m = 1; m <= 3; m++) { amp.push(rng.n() * 0.6); ph.push(2 * Math.PI * rng.u()); }
        for (i = 0; i < N; i++) {
          var v = 0;
          for (m = 1; m <= 3; m++) v += amp[m - 1] * Math.cos(2 * Math.PI * m * i / N + ph[m - 1]);
          x.push(v);
        }
        var a = [];
        for (i = 0; i < N; i++) { var s = 0; for (j = 0; j < N; j++) s += K[i][j] * x[j]; a.push(-s); }
        var sc = 0; for (i = 0; i < N; i++) sc += a[i] * a[i];
        sc = Math.sqrt(sc / N) || 1;
        for (i = 0; i < N; i++) {
          X[i * din + 0] = x[i];
          X[i * din + 1] = Math.cos(2 * Math.PI * i / N);
          X[i * din + 2] = Math.sin(2 * Math.PI * i / N);
        }
        var y = new Float64Array(N);
        for (i = 0; i < N; i++) y[i] = a[i] / sc;
        return {X: X, y: y, x: x, T: N};
      }
    };
  }

  return {
    leaf: leaf, reset: reset, backward: backward,
    mm: mm, mmNT: mmNT, softmaxRows: softmaxRows, addb: addb, add: add,
    scale: scale, tanhx: tanhx, layernorm: layernorm, mse: mse, bce: bce, row: row,
    addS: addS, sumS: sumS,
    Rng: Rng, Model: Model, Mlp: Mlp, Adam: Adam, makeTask: makeTask, makeJets: makeJets, makeRing: makeRing
  };
})();

if (typeof module !== 'undefined') module.exports = TFM;
