/* ---------------------------------------------------------------------------
   A language model small enough to train in a browser tab, and the physics it
   is asked to learn.

   Built on the tape in tfm.js: the same reverse-mode automatic differentiation,
   the same attention, plus the three things a *language* model needs that the
   reader of a set does not --

     an embedding table   - tokens are integers, so the first layer is a lookup
     a softmax over the vocabulary, with cross-entropy
     autoregressive generation - append your own output and repeat.

   The physics is a mass on a spring, sampled at even intervals and rounded
   into a vocabulary of position bins.  The exact motion then obeys an exactly
   linear two-term recurrence, which gives us something the model can be
   checked against to any number of digits.
   --------------------------------------------------------------------------- */
var LM = (function (T) {

  /* =======================================================================
     Extra tape operations
     ======================================================================= */

  /* Rows of X are gathered from an embedding table E by integer index.
     The backward pass scatters the gradient back into the used rows. */
  function embed(E, idx, n, d) {
    var out = new Float64Array(n * d), i, j;
    for (i = 0; i < n; i++)
      for (j = 0; j < d; j++) out[i * d + j] = E.v[idx[i] * d + j];
    var node = T.leaf(out, n, d);          /* placeholder; replaced below */
    return mkNode(out, n, d, function (t) {
      for (var i = 0; i < n; i++)
        for (var j = 0; j < d; j++) E.g[idx[i] * d + j] += t.g[i * d + j];
    });
  }

  /* tfm.js keeps its tape private, so reach it through a public op: build the
     node by adding a zero matrix of the right shape to a leaf.  Cheap, and it
     keeps this file from needing its own tape. */
  function mkNode(vals, n, d, bw) {
    var zero = T.leaf(new Float64Array(n * d), n, d);
    var lf = T.leaf(vals, n, d);
    var node = T.add(lf, zero);            /* on the tape, gradient -> lf.g   */
    var prev = node.bw;
    node.bw = function (t) {
      prev(t);                             /* fills lf.g and zero.g           */
      bw({g: lf.g});                        /* then our own scatter            */
    };
    return node;
  }

  /* Softmax cross-entropy over the rows of Z [n x V] against integer targets.
     Returns mean over rows, in nats. */
  function ce(Z, targets) {
    var n = Z.n, V = Z.m, loss = 0, P = new Float64Array(n * V), i, j;
    for (i = 0; i < n; i++) {
      var mx = -Infinity;
      for (j = 0; j < V; j++) if (Z.v[i * V + j] > mx) mx = Z.v[i * V + j];
      var s = 0;
      for (j = 0; j < V; j++) { P[i * V + j] = Math.exp(Z.v[i * V + j] - mx); s += P[i * V + j]; }
      for (j = 0; j < V; j++) P[i * V + j] /= s;
      loss += -Math.log(Math.max(1e-300, P[i * V + targets[i]]));
    }
    var out = new Float64Array(1); out[0] = loss / n;
    var node = T.leaf(out, 1, 1);
    /* put it on the tape by the same trick */
    var z0 = T.leaf(new Float64Array(1), 1, 1);
    var s2 = T.add(node, z0);
    var prev = s2.bw;
    s2.bw = function (t) {
      prev(t);
      for (var i = 0; i < n; i++)
        for (var j = 0; j < V; j++)
          Z.g[i * V + j] += node.g[0] * (P[i * V + j] - (j === targets[i] ? 1 : 0)) / n;
    };
    s2.probs = P;
    return s2;
  }

  /* =======================================================================
     The model: embedding + position, L blocks, a softmax over the vocabulary
     ======================================================================= */
  function GPT(opt) {
    opt = opt || {};
    var V = opt.vocab || 48, K = opt.ctx || 8, d = opt.dmodel || 32;
    var dk = opt.dk || 16, dff = opt.dff || 48, L = opt.layers || 1;
    var rng = new T.Rng(opt.seed || 5), i;
    this.V = V; this.K = K; this.d = d; this.dk = dk; this.L = L;
    function M(a, b, s) {
      var W = new Float64Array(a * b);
      for (var k = 0; k < W.length; k++) W[k] = rng.n() * s;
      return T.leaf(W, a, b);
    }
    function Vec(b, val) {
      var x = new Float64Array(b);
      if (val !== undefined) for (var k = 0; k < b; k++) x[k] = val;
      return T.leaf(x, 1, b);
    }
    this.Wemb = M(V, d, 0.4);                 /* the embedding table       */
    this.Wpos = M(K, d, 0.1);                 /* one vector per position   */
    this.blocks = [];
    for (i = 0; i < L; i++) {
      this.blocks.push({
        g1: Vec(d, 1), b1: Vec(d),
        Wq: M(d, dk, Math.sqrt(1 / d)), Wk: M(d, dk, Math.sqrt(1 / d)),
        Wv: M(d, d, Math.sqrt(1 / d)),
        g2: Vec(d, 1), b2: Vec(d),
        W1: M(d, dff, Math.sqrt(1 / d)), bf1: Vec(dff),
        W2: M(dff, d, Math.sqrt(1 / dff)), bf2: Vec(d)
      });
    }
    this.gf = Vec(d, 1); this.bf = Vec(d);
    this.Wout = M(d, V, Math.sqrt(1 / d)); this.bout = Vec(V);
    this.lastAttn = null;
  }
  GPT.prototype.params = function () {
    var p = [this.Wemb, this.Wpos], i;
    for (i = 0; i < this.blocks.length; i++) {
      var b = this.blocks[i];
      p.push(b.g1, b.b1, b.Wq, b.Wk, b.Wv, b.g2, b.b2, b.W1, b.bf1, b.W2, b.bf2);
    }
    p.push(this.gf, this.bf, this.Wout, this.bout);
    return p;
  };
  GPT.prototype.count = function () {
    var n = 0, p = this.params();
    for (var i = 0; i < p.length; i++) n += p[i].v.length;
    return n;
  };

  /* idx: array of n token indices (n <= K).  Returns logits [n x V]. */
  GPT.prototype.forward = function (idx) {
    var n = idx.length, d = this.d, i, j;
    /* embedding + position, assembled as one matrix, then put on the tape */
    var x = new Float64Array(n * d);
    for (i = 0; i < n; i++)
      for (j = 0; j < d; j++) x[i * d + j] = this.Wemb.v[idx[i] * d + j] + this.Wpos.v[i * d + j];
    var self = this;
    var h = mkNode(x, n, d, function (t) {
      for (var i = 0; i < n; i++)
        for (var j = 0; j < d; j++) {
          self.Wemb.g[idx[i] * d + j] += t.g[i * d + j];
          self.Wpos.g[i * d + j] += t.g[i * d + j];
        }
    });
    var causal = function (a, b) { return b <= a; };
    for (var l = 0; l < this.L; l++) {
      var B = this.blocks[l];
      var hn = T.layernorm(h, B.g1, B.b1);
      var Q = T.mm(hn, B.Wq), Kk = T.mm(hn, B.Wk), Vv = T.mm(hn, B.Wv);
      var S = T.scale(1 / Math.sqrt(this.dk), T.mmNT(Q, Kk));
      var A = T.softmaxRows(S, causal);
      if (l === this.L - 1) this.lastAttn = A;
      h = T.add(h, T.mm(A, Vv));
      var h2 = T.layernorm(h, B.g2, B.b2);
      h = T.add(h, T.mm(T.tanhx(T.addb(T.mm(h2, B.W1), B.bf1)), B.W2));
      h = T.addb(h, B.bf2);
    }
    h = T.layernorm(h, this.gf, this.bf);
    return T.addb(T.mm(h, this.Wout), this.bout);
  };

  /* the model's own one-step map, as a probability row over the vocabulary */
  GPT.prototype.next = function (idx) {
    T.reset();
    var Z = this.forward(idx), V = this.V, n = idx.length;
    var row = new Float64Array(V), j, mx = -Infinity, s = 0;
    for (j = 0; j < V; j++) { row[j] = Z.v[(n - 1) * V + j]; if (row[j] > mx) mx = row[j]; }
    for (j = 0; j < V; j++) { row[j] = Math.exp(row[j] - mx); s += row[j]; }
    for (j = 0; j < V; j++) row[j] /= s;
    T.reset();
    return row;
  };

  /* =======================================================================
     The physics: a mass on a spring, and the rule its samples obey
     ======================================================================= */

  /* x(t) = A cos(w t + phi), sampled every dt.  Because
        x_{n+1} + x_{n-1} = 2 cos(w dt) x_n
     holds identically for a cosine, the sampled sequence obeys an exactly
     linear two-term recurrence.  That is the rule the model has to find. */
  function Osc(opt) {
    opt = opt || {};
    var w = opt.w || 1, dt = opt.dt || 0.2, xmax = opt.xmax || 1.3;
    var V = opt.vocab || 48;
    var c2 = 2 * Math.cos(w * dt);
    return {
      w: w, dt: dt, xmax: xmax, V: V, c2: c2,
      perStep: 2 * Math.PI / (w * dt),                 /* samples per period */
      /* bin centres span [-xmax, xmax] */
      bin: function (x) {
        var k = Math.round((x + xmax) / (2 * xmax) * (V - 1));
        return k < 0 ? 0 : (k > V - 1 ? V - 1 : k);
      },
      val: function (k) { return -xmax + 2 * xmax * k / (V - 1); },
      binw: function () { return 2 * xmax / (V - 1); },
      /* a trajectory of n samples from a random amplitude and phase */
      traj: function (n, rng, A, ph) {
        A = (A === undefined) ? 0.35 + 0.6 * rng.u() : A;
        ph = (ph === undefined) ? 2 * Math.PI * rng.u() : ph;
        var x = [], i;
        for (i = 0; i < n; i++) x.push(A * Math.cos(w * dt * i + ph));
        return {x: x, A: A, ph: ph};
      },
      tokens: function (xs) { var o = [], i; for (i = 0; i < xs.length; i++) o.push(this.bin(xs[i])); return o; }
    };
  }

  /* leapfrog / Stormer-Verlet for x'' = -w^2 x.  Symplectic: the discrete
     invariant is conserved exactly, whatever the step size. */
  function leapfrog(x0, v0, w, dt, n) {
    var x = x0, v = v0, out = [x], i;
    for (i = 0; i < n; i++) {
      v += -w * w * x * dt / 2;
      x += v * dt;
      v += -w * w * x * dt / 2;
      out.push(x);
    }
    return out;
  }

  /* explicit Euler, for contrast: it gains energy and blows up */
  function euler(x0, v0, w, dt, n) {
    var x = x0, v = v0, out = [x], i;
    for (i = 0; i < n; i++) { var xn = x + v * dt; v = v - w * w * x * dt; x = xn; out.push(x); }
    return out;
  }

  /* the exact recurrence, run forward */
  function recur(x0, x1, c2, n) {
    var out = [x0, x1], i;
    for (i = 2; i <= n; i++) out.push(c2 * out[i - 1] - out[i - 2]);
    return out;
  }

  /* -----------------------------------------------------------------------
     The amplitude of a sampled cosine, from three consecutive samples.
     For x_n = A cos(w n dt + phi) the combination
         Q = x_n^2 + x_{n+1}^2 - 2 cos(w dt) x_n x_{n+1}
     equals A^2 sin^2(w dt) and is exactly constant along the true motion.
     This is the discrete stand-in for the energy.
     ----------------------------------------------------------------------- */
  function amplitude(xa, xb, c2, w, dt) {
    var c = c2 / 2;
    var Q = xa * xa + xb * xb - 2 * c * xa * xb;
    var s = Math.sin(w * dt);
    return Math.sqrt(Math.max(0, Q)) / Math.abs(s);
  }

  /* the eigenvalue modulus of the one-step matrix [[a, b],[1, 0]] */
  function lambdaMod(a, b) {
    var disc = a * a + 4 * b;                  /* x_{n+1} = a x_n + b x_{n-1} */
    if (disc < 0) return Math.sqrt(-b);        /* complex pair: |lambda|^2 = -b */
    var r1 = Math.abs((a + Math.sqrt(disc)) / 2), r2 = Math.abs((a - Math.sqrt(disc)) / 2);
    return Math.max(r1, r2);
  }

  /* least squares fit of  x_{n+1} = a x_n + b x_{n-1}  to a sequence */
  function fitRecur(xs) {
    var s11 = 0, s12 = 0, s22 = 0, t1 = 0, t2 = 0, i;
    for (i = 1; i < xs.length - 1; i++) {
      var u = xs[i], v = xs[i - 1], y = xs[i + 1];
      s11 += u * u; s12 += u * v; s22 += v * v; t1 += u * y; t2 += v * y;
    }
    var det = s11 * s22 - s12 * s12;
    if (Math.abs(det) < 1e-18) return {a: NaN, b: NaN};
    return {a: (t1 * s22 - t2 * s12) / det, b: (t2 * s11 - t1 * s12) / det};
  }

  return {
    GPT: GPT, ce: ce, Osc: Osc,
    leapfrog: leapfrog, euler: euler, recur: recur,
    amplitude: amplitude, lambdaMod: lambdaMod, fitRecur: fitRecur
  };
})(typeof TFM !== 'undefined' ? TFM : require('./tfm.js'));

if (typeof module !== 'undefined') module.exports = LM;
