/* Small plotting and signal helpers shared by the time-series demos.
   Everything is plain canvas 2-D; no libraries. */
var PL = (function () {
  var C = {ink:'#0b2559', ink2:'#27406e', muted:'#6a7c9c', rule:'#ccd8ec', panel:'#f7faff',
           blue:'#1450c8', red:'#c2273d', green:'#0b7a5a', orange:'#b4530a',
           purple:'#6b3fd4', pink:'#b3246b', teal:'#0e7490', gold:'#b8860b'};

  /* ---------- random numbers (reproducible) ---------- */
  function RNG(seed) {
    var s = seed >>> 0 || 12345;
    this.u = function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    this.n = function () { var a = this.u() || 1e-12, b = this.u();
      return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); };
  }

  /* ---------- a framed panel with axes ---------- */
  function Axes(ctx, x, y, w, h, opt) {
    opt = opt || {};
    this.ctx = ctx; this.x = x; this.y = y; this.w = w; this.h = h;
    this.xlim = opt.xlim || [0, 1]; this.ylim = opt.ylim || [-1, 1];
    this.logx = !!opt.logx; this.logy = !!opt.logy;
    ctx.fillStyle = opt.bg || C.panel; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
    if (opt.title) { ctx.fillStyle = opt.titleColor || C.ink2; ctx.font = 'bold 15px sans-serif';
      ctx.fillText(opt.title, x + 8, y - 8); }
    if (opt.xlabel) { ctx.fillStyle = C.muted; ctx.font = '13px sans-serif';
      ctx.fillText(opt.xlabel, x + w / 2 - ctx.measureText(opt.xlabel).width / 2, y + h + 36); }
    if (opt.ylabel) { ctx.fillStyle = C.muted; ctx.font = '13px sans-serif';
      ctx.fillText(opt.ylabel, x + 4, y - 8); }
  }
  Axes.prototype.X = function (v) {
    var a = this.xlim[0], b = this.xlim[1];
    if (this.logx) { v = Math.log10(Math.max(v, 1e-30)); a = Math.log10(a); b = Math.log10(b); }
    return this.x + (v - a) / (b - a) * this.w;
  };
  Axes.prototype.Y = function (v) {
    var a = this.ylim[0], b = this.ylim[1];
    if (this.logy) { v = Math.log10(Math.max(v, 1e-30)); a = Math.log10(a); b = Math.log10(b); }
    return this.y + this.h - (v - a) / (b - a) * this.h;
  };
  Axes.prototype.clip = function (f) {
    var c = this.ctx; c.save(); c.beginPath(); c.rect(this.x, this.y, this.w, this.h); c.clip(); f(); c.restore();
  };
  Axes.prototype.line = function (xs, ys, col, lw) {
    var c = this.ctx, self = this;
    this.clip(function () {
      c.strokeStyle = col; c.lineWidth = lw || 2; c.beginPath();
      for (var i = 0; i < xs.length; i++) {
        var X = self.X(xs[i]), Y = self.Y(ys[i]);
        if (i === 0) c.moveTo(X, Y); else c.lineTo(X, Y);
      }
      c.stroke();
    });
  };
  Axes.prototype.dots = function (xs, ys, col, r) {
    var c = this.ctx, self = this;
    this.clip(function () { c.fillStyle = col;
      for (var i = 0; i < xs.length; i++) { c.beginPath(); c.arc(self.X(xs[i]), self.Y(ys[i]), r || 2.5, 0, 7); c.fill(); } });
  };
  Axes.prototype.bars = function (xs, ys, col, bw) {
    var c = this.ctx, self = this;
    this.clip(function () { c.fillStyle = col;
      for (var i = 0; i < xs.length; i++) {
        var X = self.X(xs[i]), Y0 = self.Y(0), Y = self.Y(ys[i]);
        c.fillRect(X - (bw || 3) / 2, Math.min(Y, Y0), bw || 3, Math.abs(Y - Y0) || 1);
      } });
  };
  Axes.prototype.hline = function (v, col, dash) {
    var c = this.ctx; c.save(); c.strokeStyle = col; c.lineWidth = 1.5;
    if (dash) c.setLineDash(dash);
    c.beginPath(); c.moveTo(this.x, this.Y(v)); c.lineTo(this.x + this.w, this.Y(v)); c.stroke(); c.restore();
  };
  Axes.prototype.vline = function (v, col, dash) {
    var c = this.ctx; c.save(); c.strokeStyle = col; c.lineWidth = 1.5;
    if (dash) c.setLineDash(dash);
    c.beginPath(); c.moveTo(this.X(v), this.y); c.lineTo(this.X(v), this.y + this.h); c.stroke(); c.restore();
  };
  Axes.prototype.band = function (x0, x1, col) {
    var c = this.ctx; c.save(); c.fillStyle = col;
    c.fillRect(this.X(x0), this.y, this.X(x1) - this.X(x0), this.h); c.restore();
  };
  Axes.prototype.ticks = function (xt, yt, fmt) {
    var c = this.ctx; c.fillStyle = C.muted; c.font = '12px ui-monospace,Menlo,monospace';
    fmt = fmt || function (v) { return (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)) ? v.toExponential(0) : String(Math.round(v * 100) / 100); };
    var i;
    if (xt) for (i = 0; i < xt.length; i++) { var X = this.X(xt[i]);
      c.strokeStyle = C.rule; c.beginPath(); c.moveTo(X, this.y + this.h); c.lineTo(X, this.y + this.h + 4); c.stroke();
      var s = fmt(xt[i]); c.fillText(s, X - c.measureText(s).width / 2, this.y + this.h + 17); }
    if (yt) for (i = 0; i < yt.length; i++) { var Y = this.Y(yt[i]);
      c.strokeStyle = C.rule; c.beginPath(); c.moveTo(this.x - 4, Y); c.lineTo(this.x, Y); c.stroke();
      var t = fmt(yt[i]); c.fillText(t, this.x - 8 - c.measureText(t).width, Y + 4); }
  };
  Axes.prototype.legend = function (items) {   // [[label,colour], ...]
    var c = this.ctx, yy = this.y + 16;
    c.font = '14px sans-serif';
    for (var i = 0; i < items.length; i++) {
      c.fillStyle = items[i][1];
      c.fillRect(this.x + this.w - 150, yy - 10, 18, 4);
      c.fillStyle = C.ink2; c.fillText(items[i][0], this.x + this.w - 126, yy - 4);
      yy += 20;
    }
  };

  /* ---------- signal helpers ---------- */
  function acf(x, maxlag) {
    var n = x.length, m = 0, i, k;
    for (i = 0; i < n; i++) m += x[i]; m /= n;
    var c0 = 0; for (i = 0; i < n; i++) c0 += (x[i] - m) * (x[i] - m);
    var out = [];
    for (k = 0; k <= maxlag; k++) {
      var s = 0;
      for (i = 0; i + k < n; i++) s += (x[i] - m) * (x[i + k] - m);
      out.push(c0 > 0 ? s / c0 : 0);
    }
    return out;
  }
  /* radix-2 FFT, in place on re/im arrays */
  function fft(re, im) {
    var n = re.length, i, j = 0, k;
    for (i = 0; i < n - 1; i++) {
      if (i < j) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      k = n >> 1;
      while (k <= j) { j -= k; k >>= 1; }
      j += k;
    }
    for (var len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (i = 0; i < n; i += len) {
        var cr = 1, ci = 0;
        for (j = 0; j < len / 2; j++) {
          var ar = re[i + j], ai = im[i + j];
          var br = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
          var bi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
          re[i + j] = ar + br; im[i + j] = ai + bi;
          re[i + j + len / 2] = ar - br; im[i + j + len / 2] = ai - bi;
          var ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }
  function periodogram(x, dt, window) {
    var n = 1; while (n * 2 <= x.length) n *= 2;          // largest power of two
    var re = new Array(n), im = new Array(n), i, w, wss = 0;
    for (i = 0; i < n; i++) {
      w = window === 'hann' ? 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1))) : 1;
      wss += w * w;
      re[i] = x[i] * w; im[i] = 0;
    }
    fft(re, im);
    var f = [], p = [];
    for (i = 1; i < n / 2; i++) {
      f.push(i / (n * dt));
      p.push(2 * dt * (re[i] * re[i] + im[i] * im[i]) / wss);
    }
    return {f: f, p: p};
  }
  function movmean(x, k) {
    var out = [], i, j, s, c;
    for (i = 0; i < x.length; i++) {
      s = 0; c = 0;
      for (j = Math.max(0, i - k); j <= Math.min(x.length - 1, i + k); j++) { s += x[j]; c++; }
      out.push(s / c);
    }
    return out;
  }
  function range(n, f) { var a = []; for (var i = 0; i < n; i++) a.push(f ? f(i) : i); return a; }

  return {C: C, RNG: RNG, Axes: Axes, acf: acf, fft: fft, periodogram: periodogram,
          movmean: movmean, range: range};
})();
