// Players for the page animations. Each runs only while on screen; none autoplays under reduced motion.
(function () {
  const S = window.Sims;
  const C = { ink: '#1c2530', mut: '#5b6b7a', line: '#e3e8ee', soft: '#f6f8fb', acc: '#3D7EBF', acc2: '#245a95', green: '#5BA552', body: '#b24a40' };
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setupCanvas(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, w, h];
  }
  function axes(g, x0, y0, w, h, o) {          // light frame, ticks and labels
    g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(x0 + .5, y0 + .5, w, h);
    g.fillStyle = C.mut; g.font = '11px -apple-system,Segoe UI,Roboto,sans-serif';
    if (o.xlabel) { g.textAlign = 'center'; g.fillText(o.xlabel, x0 + w / 2, y0 + h + 28); }
    if (o.ylabel) { g.save(); g.translate(x0 - 30, y0 + h / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(o.ylabel, 0, 0); g.restore(); }
    (o.xt || []).forEach(([v, s]) => { g.textAlign = 'center'; g.fillText(s, x0 + v * w, y0 + h + 14); });
    (o.yt || []).forEach(([v, s]) => { g.textAlign = 'right'; g.fillText(s, x0 - 5, y0 + h - v * h + 4); });
  }

  // generic player: play/pause, reset, runs only while visible
  function player(root, sim) {
    const btn = root.querySelector('[data-play]'), rst = root.querySelector('[data-reset]');
    let running = false, visible = false, raf = 0;
    const loop = () => { raf = 0; if (!running || !visible) return; const more = sim.tick(); sim.draw(); if (more === false) { setRun(false); return; } raf = requestAnimationFrame(loop); };
    const setRun = (r) => { running = r; btn.textContent = r ? 'Pause' : 'Play'; btn.setAttribute('aria-pressed', r); if (r && visible && !raf) raf = requestAnimationFrame(loop); };
    btn.addEventListener('click', () => { if (!running && sim.done && sim.done()) { sim.reset(); } setRun(!running); });
    rst.addEventListener('click', () => { sim.reset(); sim.draw(); });
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible && running && !raf) raf = requestAnimationFrame(loop); }, { threshold: 0.15 }).observe(root);
    window.addEventListener('resize', () => sim.draw());
    sim.draw(); setRun(!reduce);
    return { restart() { sim.reset(); sim.draw(); setRun(true); } };
  }

  // ---------- two-stream ----------
  function twoStream(root) {
    const cv = root.querySelector('canvas.main'), cg = root.querySelector('canvas.side'), out = root.querySelector('[data-out]');
    const sl = root.querySelector('input[name=v0]'), slv = root.querySelector('[data-v0]');
    let ts = new S.TwoStream({ v0: +sl.value }), frame = 0;
    const sim = {
      reset() { ts = new S.TwoStream({ v0: +sl.value }); frame = 0; },
      done: () => ts.t >= 60,
      tick() { frame++; if (frame % 2 === 0) ts.step(); return ts.t < 60; },
      draw() {
        let [g, w, h] = setupCanvas(cv); const L = ts.L, vmax = 3, x0 = 40, y0 = 8, pw = w - x0 - 10, ph = h - y0 - 36;
        g.clearRect(0, 0, w, h); g.fillStyle = '#fff'; g.fillRect(x0, y0, pw, ph);
        for (let b = 0; b < 2; b++) { g.fillStyle = b ? 'rgba(61,126,191,.55)' : 'rgba(91,165,82,.55)';
          for (let i = b; i < ts.N; i += 2) { const px = x0 + ts.x[i] / L * pw, py = y0 + ph / 2 - ts.v[i] / vmax * ph / 2;
            if (py > y0 && py < y0 + ph) g.fillRect(px, py, 1.3, 1.3); } }
        axes(g, x0, y0, pw, ph, { xlabel: 'position x', ylabel: 'velocity v', xt: [[0, '0'], [.5, 'L/2'], [1, 'L']], yt: [[0, '−3'], [.5, '0'], [1, '3']] });
        // field-energy history, log scale
        [g, w, h] = setupCanvas(cg); g.clearRect(0, 0, w, h); const X0 = 44, Y0 = 8, W = w - X0 - 8, H = h - Y0 - 36;
        const lo = -5, hi = 1, tmax = 60;
        g.strokeStyle = C.line; [-4, -2, 0].forEach((e) => { const y = Y0 + H - (e - lo) / (hi - lo) * H; g.beginPath(); g.moveTo(X0, y); g.lineTo(X0 + W, y); g.stroke(); });
        g.strokeStyle = C.acc2; g.lineWidth = 2; g.beginPath();
        ts.Ehist.forEach(([t, W_], i) => { const y = Y0 + H - (Math.log10(Math.max(W_, 1e-5)) - lo) / (hi - lo) * H, x = X0 + t / tmax * W; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
        axes(g, X0, Y0, W, H, { xlabel: 'time (1/ω_p)', ylabel: 'field energy', xt: [[0, '0'], [.5, '30'], [1, '60']], yt: [[1 / 6, '10⁻⁴'], [.5, '10⁻²'], [5 / 6, '1']] });
        out.textContent = `t = ${ts.t.toFixed(1)} ω_p⁻¹`;
      }
    };
    sl.addEventListener('input', () => { slv.textContent = (+sl.value).toFixed(1); });
    const p = player(root, sim); sl.addEventListener('change', () => p.restart());
  }

  // ---------- dynamical friction ----------
  function friction(root) {
    const cvs = root.querySelectorAll('canvas.main'), cr = root.querySelector('canvas.side'), out = root.querySelector('[data-out]');
    const eps = [0.02, 0.5], N = 3000; let F, trails;
    const sim = {
      reset() { F = eps.map((e) => new S.Friction({ eps: e, N, Mb: 0.04 })); trails = [[], []]; },
      done: () => F[0].t >= 60,
      tick() { F.forEach((f, i) => { f.step(3); trails[i].push([f.b.x, f.b.y]); if (trails[i].length > 160) trails[i].shift(); }); return F[0].t < 60; },
      draw() {
        F.forEach((f, idx) => {
          const [g, w, h] = setupCanvas(cvs[idx]); const s = Math.min(w, h) / 6.4, cx = w / 2, cy = h / 2;
          g.fillStyle = '#fff'; g.fillRect(0, 0, w, h); g.fillStyle = 'rgba(36,90,149,.33)';
          for (let i = 0; i < f.N; i++) { const px = cx + f.x[i] * s, py = cy - f.y[i] * s; if (px > 0 && px < w && py > 0 && py < h) g.fillRect(px, py, 1.4, 1.4); }
          g.strokeStyle = 'rgba(178,74,64,.45)'; g.lineWidth = 1.5; g.beginPath();
          trails[idx].forEach(([x, y], i) => { const px = cx + x * s, py = cy - y * s; i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
          g.fillStyle = C.body; g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(cx + f.b.x * s, cy - f.b.y * s, 5, 0, 7); g.fill(); g.stroke();
          g.fillStyle = C.ink; g.font = '12px -apple-system,Segoe UI,Roboto,sans-serif'; g.textAlign = 'left';
          g.fillText(idx ? 'ε = 0.5 (soft)' : 'ε = 0.02 (hard)', 8, 16);
          g.strokeStyle = C.line; g.strokeRect(.5, .5, w - 1, h - 1);
        });
        const [g, w, h] = setupCanvas(cr); g.clearRect(0, 0, w, h); const X0 = 40, Y0 = 8, W = w - X0 - 70, H = h - Y0 - 36, tmax = 60, rmax = 3;
        g.strokeStyle = C.line; [1, 2].forEach((r) => { const y = Y0 + H - r / rmax * H; g.beginPath(); g.moveTo(X0, y); g.lineTo(X0 + W, y); g.stroke(); });
        const cols = [C.acc2, '#8fb0d6'];
        F.forEach((f, i) => { g.strokeStyle = cols[i]; g.lineWidth = 2; g.beginPath();
          f.rhist.forEach(([t, r], j) => { const x = X0 + t / tmax * W, y = Y0 + H - Math.min(r, rmax) / rmax * H; j ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
          const [t, r] = f.rhist[f.rhist.length - 1]; g.fillStyle = C.ink; g.font = '11px -apple-system,Segoe UI,Roboto,sans-serif'; g.textAlign = 'left';
          g.fillText('ε = ' + eps[i], X0 + t / tmax * W + 6, Y0 + H - Math.min(r, rmax) / rmax * H + (i ? -4 : 12)); });
        axes(g, X0, Y0, W, H, { xlabel: 'time', ylabel: 'orbit radius', xt: [[0, '0'], [.5, '30'], [1, '60']], yt: [[0, '0'], [1 / 3, '1'], [2 / 3, '2'], [1, '3']] });
        out.textContent = `t = ${F[0].t.toFixed(1)}`;
      }
    };
    sim.reset(); player(root, sim);
  }

  // ---------- Orszag-Tang ----------
  function cmapMagma(t) { // compact 5-stop approximation of magma
    const st = [[0, [10, 7, 34]], [.25, [81, 18, 124]], [.5, [183, 55, 121]], [.75, [252, 137, 97]], [1, [252, 253, 191]]];
    t = Math.max(0, Math.min(1, t)); let i = 0; while (i < 3 && t > st[i + 1][0]) i++;
    const f = (t - st[i][0]) / (st[i + 1][0] - st[i][0]); return st[i][1].map((a, k) => a + f * (st[i + 1][1][k] - a));
  }
  function orszag(root) {
    const cv = root.querySelector('canvas.main'), out = root.querySelector('[data-out]'), sel = root.querySelector('select');
    const n = window.innerWidth < 560 ? 96 : 128; let o = new S.OrszagTang({ n });
    const off = document.createElement('canvas'); off.width = n; off.height = n; const og = off.getContext('2d'), img = og.createImageData(n, n);
    const sim = {
      reset() { o = new S.OrszagTang({ n }); },
      done: () => o.t >= 3.14,
      tick() { o.step(); return o.t < 3.14; },
      draw() {
        const q = sel.value; let arr, lo, hi;
        if (q === 'rho') { arr = o.U[0]; lo = 1; hi = 6.3; }
        else { const [bx, by] = o.B(o.A); arr = new Float64Array(n * n); for (let k = 0; k < n * n; k++) arr[k] = 0.5 * (bx[k] * bx[k] + by[k] * by[k]); lo = 0; hi = 3; }
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const c = cmapMagma((arr[j * n + i] - lo) / (hi - lo)), p = 4 * ((n - 1 - j) * n + i);
          img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255; }
        og.putImageData(img, 0, 0);
        const [g, w, h] = setupCanvas(cv); g.imageSmoothingEnabled = true; g.clearRect(0, 0, w, h); const s = Math.min(w, h); g.drawImage(off, (w - s) / 2, 0, s, s);
        out.textContent = `t = ${o.t.toFixed(2)} · grid ${n}² · ∇·B = 0 by construction`;
      }
    };
    sel.addEventListener('change', () => sim.draw()); player(root, sim);
  }

  // ---------- CMB ----------
  function cmapDiv(t) { // blue - neutral grey - red
    t = Math.max(-1, Math.min(1, t)); const a = [43, 108, 176], m = [238, 238, 236], b = [192, 57, 43];
    return t < 0 ? m.map((v, k) => v + (-t) * (a[k] - v)) : m.map((v, k) => v + t * (b[k] - v));
  }
  function cmb(root) {
    const cv = root.querySelector('canvas.main'), cs = root.querySelector('canvas.side'), out = root.querySelector('[data-out]');
    const sl = root.querySelector('input[name=R]'), slv = root.querySelector('[data-R]');
    const c = new S.CMB({ R: +sl.value }); const n = c.n; let eta = 0.02, hold = 0;
    const off = document.createElement('canvas'); off.width = n; off.height = n; const og = off.getContext('2d'), img = og.createImageData(n, n);
    let scale = 1; { const m = c.map(1); let s2 = 0; for (const v of m) s2 += v * v; scale = 2.6 * Math.sqrt(s2 / m.length); }
    const sim = {
      reset() { eta = 0.02; hold = 0; },
      done: () => hold > 90,
      tick() { if (eta < 1) eta = Math.min(1, eta + 0.004); else hold++; return hold <= 90; },
      draw() {
        const m = c.map(eta);
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const col = cmapDiv(m[j * n + i] / scale), p = 4 * (j * n + i);
          img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255; }
        og.putImageData(img, 0, 0);
        let [g, w, h] = setupCanvas(cv); g.clearRect(0, 0, w, h); const s = Math.min(w, h); g.imageSmoothingEnabled = true; g.drawImage(off, (w - s) / 2, 0, s, s);
        [g, w, h] = setupCanvas(cs); g.clearRect(0, 0, w, h); const X0 = 44, Y0 = 8, W = w - X0 - 10, H = h - Y0 - 36, kmax = 45, ymax = 2.2;
        g.strokeStyle = C.line; [0.5, 1, 1.5, 2].forEach((v) => { const y = Y0 + H - v / ymax * H; g.beginPath(); g.moveTo(X0, y); g.lineTo(X0 + W, y); g.stroke(); });
        g.strokeStyle = C.acc2; g.lineWidth = 2; g.beginPath();
        for (let k = 1; k <= kmax; k += 0.25) { const T = c.transfer(k, eta), x = X0 + k / kmax * W, y = Y0 + H - Math.min(T * T, ymax) / ymax * H; k === 1 ? g.moveTo(x, y) : g.lineTo(x, y); }
        g.stroke();
        axes(g, X0, Y0, W, H, { xlabel: 'wavenumber k  (→ multipole ℓ on the sky)', ylabel: 'power  |Θ₀+Ψ|²', xt: [[0, '0'], [1 / 3, '15'], [2 / 3, '30'], [1, '45']], yt: [[0, '0'], [1 / 2.2, '1'], [2 / 2.2, '2']] });
        if (eta >= 1) { g.fillStyle = C.ink; g.font = '600 12px -apple-system,Segoe UI,Roboto,sans-serif'; g.textAlign = 'right'; g.fillText('recombination: pattern frozen', X0 + W - 6, Y0 + 16); }
        out.textContent = eta < 1 ? `η / η* = ${eta.toFixed(2)} · tightly coupled fluid` : 'η = η* · photons now free-stream to us';
      }
    };
    const p = player(root, sim);
    sl.addEventListener('input', () => { c.R = +sl.value; slv.textContent = c.R.toFixed(2); { const m = c.map(1); let s2 = 0; for (const v of m) s2 += v * v; scale = 2.6 * Math.sqrt(s2 / m.length); } sim.draw(); });
  }

  // ---------- PINN training replay ----------
  function pinn(root) {
    const D = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
    const cv = root.querySelector('canvas.main'), cn = root.querySelector('canvas.side'), out = root.querySelector('[data-out]'), sc = root.querySelector('input[name=it]');
    sc.max = D.snaps.length - 1; let i = 0, f = 0; const cols = ['#7aaee0', '#3D7EBF', '#1d4677'], tl = ['0.25', '0.5', '1'];
    const sim = {
      reset() { i = 0; f = 0; sc.value = 0; },
      done: () => i >= D.snaps.length - 1,
      tick() { if (++f % 8 === 0 && i < D.snaps.length - 1) { i++; sc.value = i; } return i < D.snaps.length - 1; },
      draw() {
        const sn = D.snaps[i]; let [g, w, h] = setupCanvas(cv); g.clearRect(0, 0, w, h);
        const X0 = 40, Y0 = 8, W = w - X0 - 10, H = h - Y0 - 36, X = (k) => X0 + k / 50 * W, Y = (u) => Y0 + H - (u + 1.1) / 2.2 * H;
        g.strokeStyle = C.line; [-1, -.5, 0, .5, 1].forEach((u) => { g.beginPath(); g.moveTo(X0, Y(u)); g.lineTo(X0 + W, Y(u)); g.stroke(); });
        D.exact.forEach((ex, s) => { g.strokeStyle = '#b8c2cc'; g.lineWidth = 1.5; g.setLineDash([4, 3]); g.beginPath(); ex.forEach((u, k) => k ? g.lineTo(X(k), Y(u)) : g.moveTo(X(k), Y(u))); g.stroke(); g.setLineDash([]);
          g.strokeStyle = cols[s]; g.lineWidth = 2; g.beginPath(); sn.u[s].forEach((u, k) => k ? g.lineTo(X(k), Y(u)) : g.moveTo(X(k), Y(u))); g.stroke(); });
        g.font = '11px -apple-system,Segoe UI,Roboto,sans-serif'; g.textAlign = 'left';
        const ly = Y0 + H - 70;
        cols.forEach((c, s) => { g.strokeStyle = c; g.lineWidth = 2; g.beginPath(); g.moveTo(X0 + 10, ly + s * 16); g.lineTo(X0 + 30, ly + s * 16); g.stroke(); g.fillStyle = C.ink; g.fillText('PINN, t = ' + tl[s], X0 + 36, ly + 4 + s * 16); });
        g.strokeStyle = '#b8c2cc'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(X0 + 10, ly + 48); g.lineTo(X0 + 30, ly + 48); g.stroke(); g.setLineDash([]); g.fillText('exact', X0 + 36, ly + 52);
        axes(g, X0, Y0, W, H, { xlabel: 'x', ylabel: 'u(t, x)', xt: [[0, '−1'], [.5, '0'], [1, '1']], yt: [[0.5, '0'], [(1 + 1.1) / 2.2, '1'], [(-1 + 1.1) / 2.2, '−1']] });
        [g, w, h] = setupCanvas(cn); g.clearRect(0, 0, w, h); const x0 = 48, y0 = 8, ww = w - x0 - 10, hh = h - y0 - 36, itmax = D.snaps[D.snaps.length - 1].it;
        const yy = (v) => y0 + hh - (v - 0.02) / 0.095 * hh, xx = (it) => x0 + it / itmax * ww;
        g.strokeStyle = C.mut; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x0, yy(0.03)); g.lineTo(x0 + ww, yy(0.03)); g.stroke(); g.setLineDash([]);
        g.fillStyle = C.mut; g.textAlign = 'right'; g.fillText('true ν = 0.030', x0 + ww, yy(0.03) + 14);
        g.strokeStyle = C.acc; g.lineWidth = 2; g.beginPath(); for (let k = 0; k <= i; k++) { const s = D.snaps[k]; k ? g.lineTo(xx(s.it), yy(s.nu)) : g.moveTo(xx(s.it), yy(s.nu)); } g.stroke();
        g.fillStyle = C.acc; g.beginPath(); g.arc(xx(sn.it), yy(sn.nu), 4, 0, 7); g.fill();
        axes(g, x0, y0, ww, hh, { xlabel: 'training iteration', ylabel: 'learned ν', xt: [[0, '0'], [3000 / itmax, '3k'], [6000 / itmax, '6k'], [1, (itmax / 1000).toFixed(0) + 'k']], yt: [[(0.03 - 0.02) / 0.095, '0.03'], [(0.06 - 0.02) / 0.095, '0.06'], [(0.09 - 0.02) / 0.095, '0.09']] });
        out.textContent = `iteration ${sn.it}${sn.it > 6000 ? ' (L-BFGS)' : ' (Adam)'} · ν = ${sn.nu.toFixed(4)}`;
      }
    };
    sc.addEventListener('input', () => { i = +sc.value; sim.draw(); });
    player(root, sim);
  }

  const run = { twostream: twoStream, friction, orszag, cmb, pinn };
  document.querySelectorAll('.anim[data-anim]').forEach((el) => { try { run[el.dataset.anim](el); } catch (e) { console.error(el.dataset.anim, e); } });
})();
