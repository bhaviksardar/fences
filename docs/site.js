// Ordered-dither fields drawn into <canvas data-dither="fence">.
// Each canvas renders at one pixel per cell and is scaled up with
// image-rendering: pixelated, so every visible square is a real 1-bit dot.

const BAYER = [
  0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
  12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
  3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
  15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
].map(n => (n + 0.5) / 64);

const CELL = 3;
const RAIL = 0.34;   // the limit: spend lines flatten here
const RAIL2 = 0.56;
const RUNS = [
  { base: 0.05, k: 1.5 }, { base: 0.38, k: 2.4 }, { base: 0.66, k: 1.1 },
  { base: 0.92, k: 1.9 }, { base: 1.2, k: 1.3 }, { base: 1.42, k: 2.8 },
];

function rgb(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const n = parseInt(v.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// Returns [intensity 0..1, tone] for a cell. Tones: 0 fence, 1 run, 2 breach, 3 haze.
const fields = {
  fence(u, v, t, aspect, el) {
    const px = 1 / el.rows;
    let i = 0, tone = 0;
    // Two rails and the posts: the fence mark, drawn as a field
    for (const r of [RAIL, RAIL2]) { const d = Math.abs(v - r); if (d < px * 1.2) i = Math.max(i, 0.5); }
    const pc = (u * el.width + t * 1.5) % 42;  // a post every 42 dots
    if (v > RAIL - 0.08 && pc < 1.6) i = Math.max(i, 0.42);
    // Ground haze
    const haze = 0.22 * smooth(0.5, 1.1, v);
    if (haze > i) { i = haze; tone = 3; }
    // Runs: spend climbs left to right until it meets the rail, then holds and stops
    for (let n = 0; n < RUNS.length; n++) {
      const run = RUNS[n];
      const u0 = (((run.base - t * 0.018) % 1.7) + 1.7) % 1.7 - 0.35;
      const x = u - u0;
      if (x < 0) continue;
      const k = run.k * (+el.dataset.slope || 1);
      const climb = 1.04 - k * x + 0.012 * Math.sin(x * 38 + n);
      const hit = (1.04 - RAIL) / k;
      if (x > hit + 0.2) continue;
      const cv = Math.max(RAIL, climb);
      const d = Math.abs(v - cv) * (x < hit ? Math.cos(Math.atan(k / aspect)) : 1);
      const w = px * 1.1;
      if (d < w * 2.2) {
        const s = d < w ? 0.95 : 0.5 * (1 - (d - w) / (w * 1.2));
        if (s > i) { i = s; tone = x >= hit && x < hit + 0.045 ? 2 : 1; }
      }
      if (x >= hit && x < hit + 0.03 && Math.abs(v - RAIL) < px * 4) { i = Math.max(i, 0.38); tone = 2; }
    }
    if (el.dataset.fade !== 'none') i *= u < 0.12 ? 0 : smooth(0.12, 0.22, u);
    return [i, tone];
  },
};

function setup(canvas) {
  const field = fields[canvas.dataset.dither];
  const ctx = canvas.getContext('2d');
  let img, colors, t = 7, raf = 0, last = 0, visible = false;

  function size() {
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width / CELL));
    canvas.height = Math.max(1, Math.round(r.height / CELL));
    canvas.rows = canvas.height;
    img = ctx.createImageData(canvas.width, canvas.height);
  }
  function theme() { colors = [rgb('--fg-3'), rgb('--fg'), rgb('--breach'), rgb('--line-2')]; }
  function draw() {
    const { width: w, height: h } = canvas, d = img.data, aspect = w / h;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const [i, tone] = field(x / w, y / h, t, aspect, canvas);
      const o = (y * w + x) * 4;
      if (i > BAYER[(y & 7) * 8 + (x & 7)]) { const c = colors[tone]; d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255; }
      else d[o + 3] = 0;
    }
    ctx.putImageData(img, 0, 0);
  }
  function loop(now) {
    raf = requestAnimationFrame(loop);
    if (now - last < 66) return; // ~15 fps is plenty for drifting dots
    t += (now - (last || now)) / 1000; last = now;
    draw();
  }
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const animated = canvas.dataset.animate !== undefined;
  function run() {
    cancelAnimationFrame(raf); last = 0;
    if (animated && visible && !still.matches) raf = requestAnimationFrame(loop); else draw();
  }
  size(); theme(); draw();
  new ResizeObserver(() => { size(); draw(); }).observe(canvas);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { theme(); draw(); });
  still.addEventListener('change', run);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; run(); }).observe(canvas);
}

document.querySelectorAll('canvas[data-dither]').forEach(setup);

