// The versus page: the upstream Lane Runner twice, the MLX GPU on the left and the Apple Neural
// Engine on the right, same seed, same barriers, started together.
//
//   /versus                   replay: benchmark run 1 of both devices, step for step (results/*.json)
//   /versus?run=3             another recorded run (every run is on disk; run 1 is the default, not a pick)
//   /versus?mode=live         live: both devices answer through apple_server.py, at the same time
//   /versus?render            a fixed 1920x1080 frame driven by tools/render_video.mjs
//
// Every number on screen comes from a laya-apple result: recorded ones in replay and render mode,
// fresh ones in live mode. The page computes nothing but smoothing and counting.
import runner from './demos/runner.js';
import { Stage } from './dither.js';
import { STEP, MAX_RATE } from './sim.js';
import { RunReplay } from './versus-core.js';
import { h } from './ui.js';

const q = new URLSearchParams(location.search);
const RENDER = q.has('render'), MODE = RENDER ? 'render' : q.get('mode') || 'replay';
const RUN = String(+(q.get('run') || 1)).padStart(3, '0');
const LANES = ['left', 'middle', 'right'], TONES = ['#0c0c0c', '#ffc609'];
const NAME = { gpu: 'MLX GPU', ane: 'Apple Neural Engine' };
const SUB = { gpu: 'MLX · FP16', ane: 'Core ML · FP16 · Neural Engine' };
const $ = s => document.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
if (RENDER) document.body.classList.add('render');

// ------------------------------------------------------------------ one side: a stage and its readout
class Side {
  constructor(el, device) {
    this.device = device;
    const dd = (k, cls) => { const d = h('dd', {}, '—'); this[k] = d; return h('div', { class: cls || '' }, h('dt', {}, cls === 'lat' ? 'ms per decision' : k), d); };
    this.canvas = h('canvas', { 'aria-label': NAME[device] + ' Lane Runner' });
    this.msg = h('div', { class: 'vs-msg' });
    this.probs = LANES.map(l => { const i = h('i'), v = h('span', {}, '—'); const row = h('div', { class: 'vs-prob' }, h('span', {}, l), i, v); return { row, i, v }; });
    el.append(
      h('div', { class: 'vs-label' }, h('b', {}, NAME[device]), h('span', {}, SUB[device])),
      h('div', { class: 'vs-stage' }, this.canvas, this.msg),
      h('dl', { class: 'vs-stats' }, dd('latency', 'lat'), dd('action'), dd('score'), dd('speed'),
        h('div', { class: 'vs-probs' }, this.probs.map(p => p.row))));
    this.stage = new Stage(this.canvas, { w: 480, h: 270, tones: TONES });
  }

  show({ inst, latest, ema, dps }) {
    inst.draw(this.stage.ctx, this.stage.w, this.stage.h); this.stage.present();
    this.msg.replaceChildren(...(inst.dead ? [h('span', {}, 'CRASH')] : []));
    this.latency.replaceChildren(latest ? ema.toFixed(1) : '—', h('small', {}, latest ? 'ms' : ''));
    this.action.textContent = latest ? latest.action.label : '—';
    this.score.textContent = String(inst.score);
    this.speed.replaceChildren(inst.speed.toFixed(1), h('small', {}, dps ? `${dps} dec/s` : ''));
    this.probs.forEach((p, k) => {
      const v = latest ? latest.probs[k] : 0;
      p.i.style.setProperty('--p', (v * 100).toFixed(1) + '%');
      p.v.textContent = latest ? (v * 100).toFixed(0) + '%' : '—';
      p.row.classList.toggle('top', !!latest && latest.answer.choice === LANES[k]);
    });
  }
}

const sides = Object.fromEntries(['gpu', 'ane'].map(d => [d, new Side(document.querySelector(`.vs-side[data-device="${d}"]`), d)]));

async function json(url) { const r = await fetch(url, { cache: 'no-store' }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }

function metaLine(env, seed) {
  return [`laya-typed-decisions`, `seed ${seed}`, env.soc, env.macos && `macOS ${env.macos}`, env.laya_apple && `laya-apple ${env.laya_apple}`].filter(Boolean).join(' · ');
}

async function loadRuns() {
  const [gpu, ane] = await Promise.all([json(`results/gpu-run-${RUN}.json`), json(`results/ane-run-${RUN}.json`)]);
  if (gpu.seed !== ane.seed || gpu.steps !== ane.steps) throw new Error('the two recorded runs do not share a seed and length');
  const summary = await json('results/summary.json').catch(() => null);
  return { gpu, ane, summary };
}

const view = r => ({ inst: r.inst, latest: r.latest, ema: r.ema, dps: r.decisionsPerSecond });

// ------------------------------------------------------------------ replay: real time, looping
async function replay() {
  const runs = await loadRuns();
  $('#meta').textContent = metaLine(runs.gpu.server, runs.gpu.seed);
  $('#source').textContent = `Recorded benchmark run ${+RUN} of ${runs.summary?.runs_per_device ?? '?'} for each device, replayed step for step. Each device was measured alone.`;
  let reps, t, last = performance.now();
  const restart = () => { reps = { gpu: new RunReplay(runs.gpu), ane: new RunReplay(runs.ane) }; t = 0; };
  restart();
  const frame = now => {
    requestAnimationFrame(frame);
    t += Math.min(0.1, (now - last) / 1000); last = now;
    const step = Math.floor(t / STEP);
    for (const d of ['gpu', 'ane']) { reps[d].seek(step); sides[d].show(view(reps[d])); }
    if (reps.gpu.done && reps.ane.done) restart();
  };
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ live: both devices at once
async function live() {
  const health = await json('api/health');
  if (!['gpu', 'ane'].every(d => health.devices?.includes(d))) throw new Error('live mode needs `apple_server.py` with both devices (the default --device both)');
  const seed = +(q.get('seed') || 20260924), params = Object.fromEntries(runner.params.map(p => [p.id, p.value]));
  $('#meta').textContent = metaLine(health, seed);
  $('#source').textContent = 'Live: both devices answer at the same time, sharing this Mac. The benchmark measures one device at a time.';
  const predict = async (obs, device) => {
    const r = await fetch('api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: obs.state, questions: obs.questions, model: 'typed-decisions', device }) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error);
    if (body.device !== device) throw new Error(`asked for ${device}, ran on ${body.device}`);
    return body;
  };
  const state = {};
  for (const d of ['gpu', 'ane']) {
    const obs = runner.create(seed).observe();
    for (let i = 0; i < 3; i++) await predict(obs, d);   // warm-up, as upstream's recorder
    state[d] = { inst: runner.create(seed), latest: null, ema: 0, n: 0, applied: [] };
  }
  let step = 0, acc = 0, last = performance.now();
  const decide = async d => {   // upstream live.js decide(): one request in flight, at most MAX_RATE a second
    const s = state[d];
    for (;;) {
      if (s.inst.dead) { await sleep(20); continue; }
      const obs = s.inst.observe(), t0 = performance.now();
      const res = await predict(obs, d);
      const probs = LANES.map(l => res.probabilities[l]);
      const answer = { choice: LANES[probs.indexOf(Math.max(...probs))], probabilities: res.probabilities };
      const action = s.inst.act({ lane: answer }, params, true);
      s.ema += (res.latency_ms - s.ema) * (s.n++ ? 0.12 : 1);
      s.latest = { ms: res.latency_ms, probs, answer, action }; s.applied.push(step);
      const rtt = performance.now() - t0;
      if (rtt < 1000 / MAX_RATE) await sleep(1000 / MAX_RATE - rtt);
    }
  };
  const frame = now => {
    requestAnimationFrame(frame);
    acc += Math.min(0.1, (now - last) / 1000); last = now;
    while (acc >= STEP) { for (const d of ['gpu', 'ane']) state[d].inst.update(STEP, {}); step++; acc -= STEP; }
    for (const d of ['gpu', 'ane']) {
      const s = state[d];
      while (s.applied.length && s.applied[0] <= step - 1 / STEP) s.applied.shift();
      sides[d].show({ inst: s.inst, latest: s.latest, ema: s.ema, dps: s.applied.length });
    }
  };
  requestAnimationFrame(frame);
  decide('gpu'); decide('ane');   // started together, from the same seed
}

// ------------------------------------------------------------------ render: one video frame at a time
// Video time -> game time. Real time, then an 8x fast-forward (labelled on screen) so the game
// visibly speeds up, then real time at the top speed, then the measured results.
const FPS = 30;
const SEGMENTS = [[0, 7, 1], [7, 11, 8], [11, 16, 1]];   // [video from, video to, game seconds per video second]
const CARD_AT = 16, END = 22;

function gameSeconds(v) {
  let g = 0;
  for (const [a, b, rate] of SEGMENTS) { if (v <= a) break; g += (Math.min(v, b) - a) * rate; }
  return g;
}

function card(runs) {
  const s = runs.summary, g = s.devices.gpu, a = s.devices.ane, ms = x => `${x.toFixed(1)} ms`;
  const median = xs => { const v = [...xs].sort((p, q) => p - q), m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2; };
  const top = d => { const m = Math.max(...d.max_speed); return m.toFixed(1); };
  const rows = [
    ['Latency P50', ms(g.latency_ms.p50), ms(a.latency_ms.p50), 'hi'],
    ['Latency P95', ms(g.latency_ms.p95), ms(a.latency_ms.p95), 'hi'],
    ['Top game speed', top(g), top(a)],
    ['Rows cleared, median run', String(median(g.rows_cleared)), String(median(a.rows_cleared))],
    ['Crashes, all runs', String(g.total_crashes), String(a.total_crashes)],
  ];
  const same = s.per_seed.filter(r => r.same_outcome).length;
  const env = s.environment.gpu;
  $('#card').replaceChildren(
    h('h2', {}, 'Measured'),
    h('table', {}, h('thead', {}, h('tr', {}, h('th'), h('th', {}, NAME.gpu), h('th', {}, NAME.ane))),
      h('tbody', {}, rows.map(([k, x, y, cls]) => h('tr', {}, h('td', {}, k), h('td', { class: cls || '' }, x), h('td', { class: cls || '' }, y))))),
    h('p', {}, `${g.runs} runs per device, the same ${g.runs} seeds, ${s.seconds} game seconds each. ${g.decisions.toLocaleString('en')} GPU and ${a.decisions.toLocaleString('en')} ANE decisions. `
      + `Latency is laya-apple's own per-call measurement. Each device measured alone. Same score and crashes on ${same} of ${s.per_seed.length} seeds.`),
    h('p', {}, [`laya-typed-decisions`, env.soc, `macOS ${env.macos}`, `laya-apple ${env.laya_apple}`, `MLX ${env.mlx}`, `coremltools ${env.coremltools}`].filter(Boolean).join(' · ')),
    h('div', { class: 'repo' }, 'laya-apple  ', h('span', {}, 'github.com/tc3oliver/laya-apple')));
}

async function render() {
  const runs = await loadRuns();
  if (!runs.summary) throw new Error('render needs results/summary.json (run summarize.py)');
  $('#meta').textContent = metaLine({ ...runs.gpu.server, soc: runs.summary.environment.gpu.soc }, runs.gpu.seed);
  $('#source').textContent = `Recorded benchmark run ${+RUN} of ${runs.summary.runs_per_device} per device, replayed. Each device measured alone.`;
  card(runs);
  const reps = { gpu: new RunReplay(runs.gpu), ane: new RunReplay(runs.ane) };
  window.vsRender = {
    fps: FPS, frames: Math.round(END * FPS),
    seek(frame) {   // frames must be requested in increasing order
      const v = frame / FPS, step = Math.round(gameSeconds(Math.min(v, CARD_AT)) / STEP);
      for (const d of ['gpu', 'ane']) { reps[d].seek(step); sides[d].show(view(reps[d])); }
      const ff = SEGMENTS.find(([a, b]) => v >= a && v < b);
      $('#rate').textContent = ff && ff[2] !== 1 && v < CARD_AT ? `▶▶ ${ff[2]}× FAST-FORWARD` : '';
      $('#card').hidden = v < CARD_AT;
      return { v, step, game_s: step * STEP };
    },
  };
  window.vsReady = true;
}

({ replay, live, render })[MODE]().catch(e => {
  $('#source').textContent = 'Error: ' + e.message;
  window.vsError = e.message;
});
