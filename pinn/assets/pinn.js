/* ---------------------------------------------------------------------------
   A miniature automatic-differentiation engine and a physics-informed network,
   in plain JavaScript. Enough that every demonstration on these pages trains a
   real PINN in the browser: no library, no pre-computed results.

   The pieces:
     * a tape-based reverse-mode autodiff over small dense matrices,
     * an MLP that carries u, du/dx and d2u/dx2 forward together, so the PDE
       residual is itself a differentiable expression in the weights,
     * Adam, and a trainer that assembles the composite loss.
   --------------------------------------------------------------------------- */
var PINN = (function () {

  /* =======================================================================
     1. The tape
     Every node holds a value array v and a gradient array g of equal length,
     plus the shape (n rows, m columns) and a backward closure. Nodes are
     created in topological order, so running the tape backwards is a valid
     reverse sweep.
     ======================================================================= */
  var TAPE = [];

  function nd(v, n, m, bw) {
    var t = {v: v, g: new Float64Array(v.length), n: n, m: m, bw: bw};
    TAPE.push(t);
    return t;
  }
  function leaf(v, n, m) { return {v: v, g: new Float64Array(v.length), n: n, m: m, bw: null}; }
  function reset() { TAPE.length = 0; }
  function backward(loss) {
    loss.g[0] = 1;
    for (var i = TAPE.length - 1; i >= 0; i--) if (TAPE[i].bw) TAPE[i].bw(TAPE[i]);
  }

  /* ---------------- operations ---------------- */

  function matmul(A, W) {                       /* A: n x p, W: p x q  ->  n x q */
    var n = A.n, p = A.m, q = W.m, out = new Float64Array(n * q), i, k, j, a;
    for (i = 0; i < n; i++)
      for (k = 0; k < p; k++) {
        a = A.v[i * p + k];
        if (a === 0) continue;
        for (j = 0; j < q; j++) out[i * q + j] += a * W.v[k * q + j];
      }
    return nd(out, n, q, function (t) {
      var i, k, j, s, g, av;
      for (i = 0; i < n; i++)
        for (k = 0; k < p; k++) {
          s = 0; av = A.v[i * p + k];
          for (j = 0; j < q; j++) {
            g = t.g[i * q + j];
            s += g * W.v[k * q + j];
            W.g[k * q + j] += av * g;
          }
          A.g[i * p + k] += s;
        }
    });
  }

  function addb(A, b) {                         /* A: n x q, b: 1 x q */
    var n = A.n, q = A.m, out = new Float64Array(n * q), i, j;
    for (i = 0; i < n; i++) for (j = 0; j < q; j++) out[i * q + j] = A.v[i * q + j] + b.v[j];
    return nd(out, n, q, function (t) {
      for (var i = 0; i < n; i++) for (var j = 0; j < q; j++) {
        A.g[i * q + j] += t.g[i * q + j];
        b.g[j] += t.g[i * q + j];
      }
    });
  }

  function tanhx(A) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = Math.tanh(A.v[i]);
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += t.g[i] * (1 - out[i] * out[i]);
    });
  }

  /* s = 1 - a^2, applied to an activation a: this is tanh'(z) written in terms
     of the output, which is what makes the derivative recursion cheap. */
  function omsq(A) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = 1 - A.v[i] * A.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += -2 * A.v[i] * t.g[i];
    });
  }

  function mul(A, B) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = A.v[i] * B.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) {
        A.g[i] += B.v[i] * t.g[i];
        B.g[i] += A.v[i] * t.g[i];
      }
    });
  }

  function axpy(a, A, b, B) {                   /* a*A + b*B */
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = a * A.v[i] + b * B.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) { A.g[i] += a * t.g[i]; B.g[i] += b * t.g[i]; }
    });
  }

  function scale(a, A) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = a * A.v[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += a * t.g[i];
    });
  }

  function addConst(A, c) {                     /* c: plain array, not trainable */
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = A.v[i] + c[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += t.g[i];
    });
  }

  function mulConst(A, c) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = A.v[i] * c[i];
    return nd(out, A.n, A.m, function (t) {
      for (var i = 0; i < out.length; i++) A.g[i] += c[i] * t.g[i];
    });
  }

  /* multiply by a trainable scalar: this is how an unknown physical constant
     enters the residual in an inverse problem */
  function mulPar(A, p) {
    var out = new Float64Array(A.v.length), i;
    for (i = 0; i < out.length; i++) out[i] = A.v[i] * p.v[0];
    return nd(out, A.n, A.m, function (t) {
      var s = 0;
      for (var i = 0; i < out.length; i++) { A.g[i] += p.v[0] * t.g[i]; s += A.v[i] * t.g[i]; }
      p.g[0] += s;
    });
  }

  /* mean of w * (A - target)^2, optionally with a per-point weight array */
  function msq(A, w, target, wArr) {
    var n = A.v.length, s = 0, i, d, ww;
    for (i = 0; i < n; i++) {
      d = A.v[i] - (target ? target[i] : 0);
      ww = wArr ? wArr[i] : 1;
      s += ww * d * d;
    }
    var out = new Float64Array(1); out[0] = w * s / n;
    return nd(out, 1, 1, function (t) {
      for (var i = 0; i < n; i++) {
        var d = A.v[i] - (target ? target[i] : 0), ww = wArr ? wArr[i] : 1;
        A.g[i] += t.g[0] * w * 2 * ww * d / n;
      }
    });
  }

  function addS(a, b) {
    var out = new Float64Array(1); out[0] = a.v[0] + b.v[0];
    return nd(out, 1, 1, function (t) { a.g[0] += t.g[0]; b.g[0] += t.g[0]; });
  }
  function sumS(list) {
    var s = list[0];
    for (var i = 1; i < list.length; i++) s = addS(s, list[i]);
    return s;
  }

  /* =======================================================================
     2. Random numbers
     ======================================================================= */
  function Rng(seed) {
    var s = (seed >>> 0) || 7;
    this.u = function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    this.n = function () {
      var a = this.u() || 1e-12, b = this.u();
      return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
    };
  }

  /* =======================================================================
     3. Input features
     The network sees t = (x - c)/s rather than x, which is the single most
     important piece of housekeeping in a PINN. Optionally t is first lifted
     into a Fourier feature vector [sin(w t), cos(w t), ...], which is the
     standard cure for spectral bias.
     ======================================================================= */
  function features(xs, ff, c, sc) {
    var n = xs.length, i, k;
    if (!ff) {
      var a = new Float64Array(n), ap = new Float64Array(n), app = new Float64Array(n);
      for (i = 0; i < n; i++) { a[i] = (xs[i] - c) / sc; ap[i] = 1 / sc; app[i] = 0; }
      return {a: leaf(a, n, 1), ap: leaf(ap, n, 1), app: leaf(app, n, 1)};
    }
    var K = ff.length, m = 2 * K;
    var A = new Float64Array(n * m), Ap = new Float64Array(n * m), App = new Float64Array(n * m);
    for (i = 0; i < n; i++) {
      var t = (xs[i] - c) / sc;
      for (k = 0; k < K; k++) {
        var w = ff[k], wx = w / sc, si = Math.sin(w * t), co = Math.cos(w * t);
        A[i * m + 2 * k]     = si;  Ap[i * m + 2 * k]     =  wx * co;  App[i * m + 2 * k]     = -wx * wx * si;
        A[i * m + 2 * k + 1] = co;  Ap[i * m + 2 * k + 1] = -wx * si;  App[i * m + 2 * k + 1] = -wx * wx * co;
      }
    }
    return {a: leaf(A, n, m), ap: leaf(Ap, n, m), app: leaf(App, n, m)};
  }

  /* =======================================================================
     4. The network
     ======================================================================= */
  function MLP(widths, opt) {
    opt = opt || {};
    var rng = new Rng(opt.seed || 1), i, k;
    this.ff = opt.fourier || null;               /* array of angular frequencies */
    this.c = (opt.domain ? (opt.domain[0] + opt.domain[1]) / 2 : 0);
    this.sc = (opt.domain ? (opt.domain[1] - opt.domain[0]) / 2 : 1);
    var w = widths.slice();
    if (this.ff) w[0] = 2 * this.ff.length;
    this.layers = [];
    for (i = 0; i + 1 < w.length; i++) {
      var fin = w[i], fout = w[i + 1], s = Math.sqrt(1 / fin);
      var W = new Float64Array(fin * fout);
      for (k = 0; k < W.length; k++) W[k] = rng.n() * s;
      this.layers.push({W: leaf(W, fin, fout), b: leaf(new Float64Array(fout), 1, fout)});
    }
    this.extra = {};                              /* named trainable scalars */
  }

  MLP.prototype.par = function (name, init) {
    if (!this.extra[name]) {
      var v = new Float64Array(1); v[0] = init;
      this.extra[name] = leaf(v, 1, 1);
    }
    return this.extra[name];
  };

  MLP.prototype.params = function () {
    var p = [], i;
    for (i = 0; i < this.layers.length; i++) { p.push(this.layers[i].W); p.push(this.layers[i].b); }
    for (var k in this.extra) if (this.extra.hasOwnProperty(k)) p.push(this.extra[k]);
    return p;
  };

  MLP.prototype.count = function () {
    var n = 0;
    for (var i = 0; i < this.layers.length; i++) n += this.layers[i].W.v.length + this.layers[i].b.v.length;
    return n;
  };

  /* Forward pass carrying the x-derivatives alongside the value.
     order 0 = value only, 1 = value and first derivative, 2 = both derivatives. */
  MLP.prototype.fwd = function (xs, order) {
    var f = features(xs, this.ff, this.c, this.sc);
    var a = f.a, ap = f.ap, app = f.app, L = this.layers, i;
    for (i = 0; i < L.length; i++) {
      var z   = addb(matmul(a, L[i].W), L[i].b);
      var zp  = (order >= 1) ? matmul(ap, L[i].W) : null;
      var zpp = (order >= 2) ? matmul(app, L[i].W) : null;
      if (i === L.length - 1) { a = z; ap = zp; app = zpp; break; }   /* linear output layer */
      var na = tanhx(z), s = omsq(na);
      var nap = (order >= 1) ? mul(s, zp) : null;
      var napp = null;
      /* a'' = s z''  -  2 a s (z')^2 , with s = 1 - a^2 */
      if (order >= 2) napp = axpy(1, mul(s, zpp), -2, mul(mul(na, s), mul(zp, zp)));
      a = na; ap = nap; app = napp;
    }
    return {u: a, up: ap, upp: app};
  };

  /* Evaluate without building a tape (for drawing the curve). */
  MLP.prototype.eval = function (xs, order) {
    reset();
    var o = this.fwd(xs, order === undefined ? 0 : order);
    var r = {u: Array.prototype.slice.call(o.u.v)};
    if (o.up)  r.up  = Array.prototype.slice.call(o.up.v);
    if (o.upp) r.upp = Array.prototype.slice.call(o.upp.v);
    reset();
    return r;
  };

  /* =======================================================================
     5. Hard boundary conditions
     u(x) = g(x) + b(x) N(x), with b vanishing exactly where the condition is
     imposed. The conditions are then satisfied identically and never appear in
     the loss at all.
     ======================================================================= */
  function hardBC(out, H) {
    var u = addConst(mulConst(out.u, H.b), H.g), up = null, upp = null;
    if (out.up) up = addConst(axpy(1, mulConst(out.u, H.bp), 1, mulConst(out.up, H.b)), H.gp);
    if (out.upp) {
      var t1 = mulConst(out.u, H.bpp);
      var t2 = scale(2, mulConst(out.up, H.bp));
      var t3 = mulConst(out.upp, H.b);
      upp = addConst(axpy(1, axpy(1, t1, 1, t2), 1, t3), H.gpp);
    }
    return {u: u, up: up, upp: upp};
  }

  /* =======================================================================
     6. Adam
     ======================================================================= */
  function Adam(params, lr) {
    this.p = params; this.lr = lr; this.t = 0;
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
     7. The trainer
     A problem is an object:
       xcol      collocation points
       residual  function(out, net) returning the residual node
       hard      optional function(xs) returning {g,gp,gpp,b,bp,bpp}
       bcs       [{x, order, target, w}]   ignored when hard is present
       data      optional {x, y, w}
       wres      weight on the residual term
       exact     optional function(x) for the reference curve
     ======================================================================= */
  function Trainer(net, P, lr) {
    this.net = net; this.P = P;
    /* a problem with unknown physical constants registers them before the
       optimiser takes its snapshot of the parameter list */
    if (P.init) P.init(net);
    this.opt = new Adam(net.params(), lr === undefined || lr === null ? 3e-3 : lr);
    this.hist = [];        /* loss history */
    this.parts = {};       /* the separate loss terms at the last step */
    this.it = 0;
  }

  Trainer.prototype.step = function () {
    var net = this.net, P = this.P, terms = [], i;
    reset();
    this.opt.zero();

    /* the physics term */
    var out = net.fwd(P.xcol, 2);
    if (P.hard) out = hardBC(out, P.hard(P.xcol));
    var r = P.residual(out, net);
    var Lr = msq(r, P.wres === undefined ? 1 : P.wres, null, P.wcol || null);
    terms.push(Lr);

    /* boundary and initial conditions, when they are imposed softly */
    var bcParts = [];
    if (!P.hard && P.bcs) {
      for (i = 0; i < P.bcs.length; i++) {
        var B = P.bcs[i];
        var ob = net.fwd(B.x, B.order || 0);
        var field = B.order === 1 ? ob.up : (B.order === 2 ? ob.upp : ob.u);
        var Lb = msq(field, B.w === undefined ? 1 : B.w, B.target || null);
        terms.push(Lb); bcParts.push(Lb);
      }
    }

    /* observed data, for inverse problems */
    var Ld = null;
    if (P.data) {
      var od = net.fwd(P.data.x, P.hard ? 2 : 0);
      if (P.hard) od = hardBC(od, P.hard(P.data.x));
      Ld = msq(od.u, P.data.w === undefined ? 1 : P.data.w, P.data.y);
      terms.push(Ld);
    }

    var total = sumS(terms);
    backward(total);
    this.opt.step();
    this.it++;

    this.parts = {
      res: Lr.v[0],
      bc: bcParts.reduce(function (s, t) { return s + t.v[0]; }, 0),
      data: Ld ? Ld.v[0] : 0,
      total: total.v[0]
    };
    this.hist.push(this.parts.total);
    reset();
    return this.parts.total;
  };

  Trainer.prototype.run = function (k) {
    var last = 0;
    for (var i = 0; i < k; i++) last = this.step();
    return last;
  };

  /* Solution curve, with the hard constraint applied if there is one. */
  Trainer.prototype.curve = function (xs) {
    var net = this.net, P = this.P;
    reset();
    var o = net.fwd(xs, P.hard ? 2 : 0);
    if (P.hard) o = hardBC(o, P.hard(xs));
    var u = Array.prototype.slice.call(o.u.v);
    reset();
    return u;
  };

  /* Pointwise PDE residual, for the residual panels. */
  Trainer.prototype.resid = function (xs) {
    var net = this.net, P = this.P;
    reset();
    var o = net.fwd(xs, 2);
    if (P.hard) o = hardBC(o, P.hard(xs));
    var r = Array.prototype.slice.call(P.residual(o, net).v);
    reset();
    return r;
  };

  /* Relative L2 error against the exact solution. */
  Trainer.prototype.err = function (xs) {
    if (!this.P.exact) return NaN;
    var u = this.curve(xs), a = 0, b = 0;
    for (var i = 0; i < xs.length; i++) {
      var e = this.P.exact(xs[i]);
      a += (u[i] - e) * (u[i] - e); b += e * e;
    }
    return Math.sqrt(a / Math.max(b, 1e-30));
  }

  /* =======================================================================
     8. Ready-made problems used by the pages
     ======================================================================= */

  function linspace(a, b, n) {
    var x = new Float64Array(n);
    for (var i = 0; i < n; i++) x[i] = a + (b - a) * i / (n - 1);
    return x;
  }

  /* Damped oscillator  u'' + 2 z w u' + w^2 u = 0,  u(0)=1, u'(0)=0 on [0,T]. */
  function oscillator(o) {
    o = o || {};
    var w = o.w === undefined ? 3 : o.w, z = o.zeta === undefined ? 0.1 : o.zeta,
        T = o.T === undefined ? 6 : o.T, N = o.n || 128, hard = o.hard !== false;
    var wd = w * Math.sqrt(Math.max(1e-12, 1 - z * z));
    var exact = function (x) {
      return Math.exp(-z * w * x) * (Math.cos(wd * x) + (z * w / wd) * Math.sin(wd * x));
    };
    var P = {
      xcol: linspace(0, T, N),
      wres: 1,
      exact: exact,
      residual: function (out) {
        return axpy(1, out.upp, 1, axpy(2 * z * w, out.up, w * w, out.u));
      }
    };
    if (hard) {
      /* u = 1 + b(x) N(x), with b(0) = b'(0) = 0 so that u(0)=1 and u'(0)=0 hold
         identically, whatever the network does.

         Which b you choose matters far more than it looks.  Near the origin the
         true solution is 1 - (w^2/2)x^2, so N(0) is pinned at -w^2/b''(0).  A
         bare x^2 makes the network span -w^2/2 near the origin and O(1/T^2) at
         the far end; (x/T)^2 is worse still, demanding N(0) = -w^2 T^2/2.  The
         saturating envelope x^2/(1+x^2) behaves like x^2 at the origin and
         tends to 1, so N stays O(1) everywhere.  Measured on
         u'' + 2(0.05)w u' + w^2 u = 0 over [0,6], 12000 Adam steps:

             envelope        w=2      w=3      w=4
             x^2             7.4%    34.5%    69.2%
             (x/T)^2         6.1%    57.4%    70.9%
             x^2/(1+x^2)     0.06%    1.5%     1.4%

         Fifty times better from one line, and nothing else changed. */
      var env = o.env || 'sat';
      P.env = env;
      P.hard = function (xs) {
        var n = xs.length, g = new Float64Array(n), gp = new Float64Array(n), gpp = new Float64Array(n),
            b = new Float64Array(n), bp = new Float64Array(n), bpp = new Float64Array(n), s = T * T, i, x, d;
        for (i = 0; i < n; i++) {
          x = xs[i]; g[i] = 1; gp[i] = 0; gpp[i] = 0;
          if (env === 'x2')        { b[i] = x * x;     bp[i] = 2 * x;     bpp[i] = 2; }
          else if (env === 'x2T')  { b[i] = x * x / s; bp[i] = 2 * x / s; bpp[i] = 2 / s; }
          else { d = 1 + x * x; b[i] = x * x / d; bp[i] = 2 * x / (d * d); bpp[i] = (2 - 6 * x * x) / (d * d * d); }
        }
        return {g: g, gp: gp, gpp: gpp, b: b, bp: bp, bpp: bpp};
      };
    } else {
      var x0 = new Float64Array(1); x0[0] = 0;
      var one = new Float64Array(1); one[0] = 1;
      P.bcs = [{x: x0, order: 0, target: one, w: o.wbc === undefined ? 1 : o.wbc},
               {x: x0, order: 1, target: null, w: o.wbc === undefined ? 1 : o.wbc}];
    }
    return P;
  }

  /* Steady heat / Poisson  -u'' = f  on [0,1] with u(0)=u(1)=0.
     Manufactured so the exact solution is sin(pi k x) + a sin(pi m x). */
  function poisson(o) {
    o = o || {};
    var k = o.k === undefined ? 1 : o.k, m = o.m === undefined ? 4 : o.m,
        a = o.a === undefined ? 0.3 : o.a, N = o.n || 128, hard = o.hard !== false;
    var exact = function (x) { return Math.sin(Math.PI * k * x) + a * Math.sin(Math.PI * m * x); };
    var f = function (x) {
      return Math.PI * Math.PI * (k * k * Math.sin(Math.PI * k * x) + a * m * m * Math.sin(Math.PI * m * x));
    };
    var xc = linspace(0, 1, N), fc = new Float64Array(N), nfc = new Float64Array(N);
    for (var i = 0; i < N; i++) { fc[i] = f(xc[i]); nfc[i] = -fc[i]; }
    var P = {
      xcol: xc, wres: o.wres === undefined ? 1 : o.wres, exact: exact, source: f,
      /* the equation is -u'' = f, so the residual is -u'' - f: it vanishes on
         the true solution.  Writing -u'' + f instead converges neatly to minus
         the right answer, which shows up as a relative error pinned at 200%. */
      residual: function (out) { return addConst(scale(-1, out.upp), nfc); }
    };
    if (hard) {
      P.hard = function (xs) {
        var n = xs.length, g = new Float64Array(n), gp = new Float64Array(n), gpp = new Float64Array(n),
            b = new Float64Array(n), bp = new Float64Array(n), bpp = new Float64Array(n);
        for (var j = 0; j < n; j++) {
          b[j] = xs[j] * (1 - xs[j]); bp[j] = 1 - 2 * xs[j]; bpp[j] = -2;
        }
        return {g: g, gp: gp, gpp: gpp, b: b, bp: bp, bpp: bpp};
      };
    } else {
      var xb = new Float64Array(2); xb[0] = 0; xb[1] = 1;
      P.bcs = [{x: xb, order: 0, target: null, w: o.wbc === undefined ? 1 : o.wbc}];
    }
    return P;
  }

  /* Inverse problem: noisy observations of a damped oscillator, with the
     damping ratio treated as an unknown to be learned alongside the solution. */
  function inverseOsc(o) {
    o = o || {};
    var w = o.w === undefined ? 3 : o.w, zTrue = o.zeta === undefined ? 0.25 : o.zeta,
        T = o.T === undefined ? 6 : o.T, nd_ = o.ndata || 24, sig = o.noise === undefined ? 0.04 : o.noise,
        N = o.n || 128, seed = o.seed || 3;
    var wd = w * Math.sqrt(1 - zTrue * zTrue);
    var exact = function (x) {
      return Math.exp(-zTrue * w * x) * (Math.cos(wd * x) + (zTrue * w / wd) * Math.sin(wd * x));
    };
    var rng = new Rng(seed);
    var xd = new Float64Array(nd_), yd = new Float64Array(nd_);
    for (var i = 0; i < nd_; i++) { xd[i] = T * (i + 0.5) / nd_; yd[i] = exact(xd[i]) + sig * rng.n(); }
    return {
      xcol: linspace(0, T, N),
      wres: o.wres === undefined ? 1 : o.wres,
      data: {x: xd, y: yd, w: o.wdata === undefined ? 20 : o.wdata},
      exact: exact, zTrue: zTrue, omega: w,
      init: function (net) { net.par('zeta', o.zinit === undefined ? 0.0 : o.zinit); },
      hard: function (xs) {
        var n = xs.length, g = new Float64Array(n), gp = new Float64Array(n), gpp = new Float64Array(n),
            b = new Float64Array(n), bp = new Float64Array(n), bpp = new Float64Array(n), j, x, d;
        for (j = 0; j < n; j++) {
          x = xs[j]; g[j] = 1; d = 1 + x * x;
          b[j] = x * x / d; bp[j] = 2 * x / (d * d); bpp[j] = (2 - 6 * x * x) / (d * d * d);
        }
        return {g: g, gp: gp, gpp: gpp, b: b, bp: bp, bpp: bpp};
      },
      residual: function (out, net) {
        var z = net.par('zeta', o.zinit === undefined ? 0.0 : o.zinit);
        /* u'' + 2 w zeta u' + w^2 u */
        return axpy(1, axpy(1, out.upp, w * w, out.u), 2 * w, mulPar(out.up, z));
      }
    };
  }

  /* A second-order finite-difference solve of -u'' = f on [0,1], u(0)=u(1)=0,
     used on the comparison page to show what a PINN is competing against. */
  function fdPoisson(f, N) {
    var h = 1 / (N + 1), i, x = new Float64Array(N), rhs = new Float64Array(N);
    for (i = 0; i < N; i++) { x[i] = (i + 1) * h; rhs[i] = f(x[i]) * h * h; }
    var c = new Float64Array(N), d = new Float64Array(N);       /* Thomas algorithm */
    c[0] = -1 / 2; d[0] = rhs[0] / 2;
    for (i = 1; i < N; i++) {
      var den = 2 + c[i - 1];
      c[i] = -1 / den;
      d[i] = (rhs[i] + d[i - 1]) / den;
    }
    var u = new Float64Array(N);
    u[N - 1] = d[N - 1];
    for (i = N - 2; i >= 0; i--) u[i] = d[i] - c[i] * u[i + 1];
    return {x: x, u: u, h: h};
  }

  return {
    leaf: leaf, reset: reset, backward: backward,
    matmul: matmul, addb: addb, tanhx: tanhx, omsq: omsq, mul: mul, axpy: axpy,
    scale: scale, addConst: addConst, mulConst: mulConst, mulPar: mulPar, msq: msq, sumS: sumS,
    MLP: MLP, Adam: Adam, Trainer: Trainer, hardBC: hardBC, Rng: Rng,
    linspace: linspace, oscillator: oscillator, poisson: poisson,
    inverseOsc: inverseOsc, fdPoisson: fdPoisson
  };
})();

if (typeof module !== 'undefined') module.exports = PINN;
