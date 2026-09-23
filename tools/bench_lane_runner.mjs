// Drive the upstream Lane Runner headlessly against apple_server.py, one device, several seeds.
// Called by benchmark_runner.py; it can also be run by hand against a server already running:
//
//     node tools/bench_lane_runner.mjs --device ane --seeds 1,2,3 --seconds 90
//
// The game is static/demos/runner.js exactly as upstream wrote it, stepped at the browser's fixed
// timestep, with the stepping rules of upstream's tools/record_run.mjs:
//   - one request in flight; the game keeps running while it is: the simulation advances by the
//     measured round trip, rounded up to whole steps, before the action is applied;
//   - at most MAX_RATE (40) decisions per game second, the browser's cap;
//   - no decision is requested while the runner is crashed.
// Every decision is a real call to laya-apple. Nothing is scripted, cached or filtered.
//
// Per run it writes results/<device>-run-NNN.json (summary plus the decision list a replay needs)
// and traces/<device>/run-NNN.jsonl.gz (one line per decision and per crash, with the full state).
import { mkdirSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import runner from '../static/demos/runner.js';
import { STEP, MIN_GAP, MAX_RATE, answersFrom } from '../static/sim.js';

const { values: args } = parseArgs({ options: {
  api: { type: 'string', default: 'http://127.0.0.1:8770' },
  device: { type: 'string' },
  model: { type: 'string', default: 'convaiinnovations/laya-typed-decisions' },
  seeds: { type: 'string', default: '20260924' },
  'first-run': { type: 'string', default: '1' },
  seconds: { type: 'string', default: '90' },
  warmup: { type: 'string', default: '50' },
  results: { type: 'string', default: fileURLToPath(new URL('../results/', import.meta.url)) },
  traces: { type: 'string', default: fileURLToPath(new URL('../traces/', import.meta.url)) },
} });
if (!['gpu', 'ane'].includes(args.device)) throw new Error('--device gpu|ane is required');

const LANES = ['left', 'middle', 'right'];
const SECONDS = +args.seconds, TOTAL = Math.round(SECONDS / STEP);
const PARAMS = Object.fromEntries(runner.params.map(p => [p.id, p.value]));   // upstream default: stay at P >= 0.25

async function predict(obs) {
  const r = await fetch(args.api + '/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: obs.state, questions: obs.questions, model: args.model, device: args.device }) });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error);
  if (body.device !== args.device) throw new Error(`asked for ${args.device}, laya-apple ran on ${body.device}`);   // never let a fallback into the data
  return body;
}

const health = await (await fetch(args.api + '/api/health')).json();
const warm = runner.create(1).observe();
for (let i = 0; i < +args.warmup; i++) await predict(warm);   // untimed: first calls pay one-off costs

mkdirSync(args.results, { recursive: true });
mkdirSync(`${args.traces}${args.device}`, { recursive: true });

const seeds = args.seeds.split(',').map(Number);
for (const [k, seed] of seeds.entries()) {
  const run = +args['first-run'] + k, tag = String(run).padStart(3, '0');
  const inst = runner.create(seed), decisions = [], lines = [], crashes = [];
  const t0wall = Date.now(), t0 = performance.now();
  let step = 0, cleared = 0, lastScore = 0, maxSpeed = inst.speed, aliveSteps = 0, dist = 0;
  const snapshot = () => ({ lane: LANES[inst.lane], x: +inst.x.toFixed(4), speed: +inst.speed.toFixed(4), score: inst.score,
    dist: +(dist + inst.dist).toFixed(3), crashes: inst.crashes, dead: inst.dead > 0,
    obstacles: inst.rows.map(r => ({ z: +r.z.toFixed(3), blocked: r.blocked.map(Number) })) });
  const advance = n => {
    for (let i = 0; i < n && step < TOTAL; i++, step++) {
      const wasDead = inst.dead > 0, crashesBefore = inst.crashes, distBefore = inst.dist;
      inst.update(STEP, {});
      if (inst.score > lastScore) cleared += inst.score - lastScore;
      lastScore = inst.score;
      if (!wasDead) { aliveSteps++; maxSpeed = Math.max(maxSpeed, inst.speed); }
      if (inst.dist < distBefore) dist += distBefore;   // reset() after a crash starts the distance again
      if (inst.crashes > crashesBefore) {
        const ev = { type: 'crash', step, t_game: +(step * STEP).toFixed(4), ...snapshot() };   // speed and score at the moment of impact
        crashes.push(ev); lines.push(ev);
      }
    }
  };

  while (step < TOTAL) {
    if (inst.dead) { advance(1); continue; }
    const obs = inst.observe(), s0 = step, before = snapshot(), t = performance.now();
    const res = await predict(obs);
    const rtt = performance.now() - t;
    advance(Math.max(1, Math.ceil(rtt / 1000 / STEP)));   // the game kept running meanwhile
    const probs = LANES.map(l => res.probabilities[l]);
    const answers = answersFrom(obs, probs);             // what a replay rebuilds, so live and replay act identically
    const action = inst.act(answers, PARAMS, true);
    decisions.push([s0, step, probs, +res.latency_ms.toFixed(3)]);
    lines.push({ type: 'decision', i: decisions.length - 1, seed, device: res.device, backend: res.backend,
      timestamp: new Date(t0wall + (t - t0)).toISOString(), t_wall_ms: +(t - t0).toFixed(3),
      step_observed: s0, step_applied: step, t_game: +(s0 * STEP).toFixed(4),
      model_input: { state: obs.state, questions: obs.questions }, model_answer: res.choice, argmax: answers.lane.choice,
      probabilities: res.probabilities, latency_ms: res.latency_ms, rtt_ms: +rtt.toFixed(3), routing_reason: res.routing_reason,
      sequence_length: res.sequence_length, action: action.label, why: action.why, state_at_observe: before, state_at_apply: snapshot() });
    if (step - s0 < MIN_GAP) advance(MIN_GAP - (step - s0));
  }
  const wall = (performance.now() - t0) / 1000;

  const lat = decisions.map(d => d[3]).sort((a, b) => a - b), q = p => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))];
  const counts = { STAY: 0, LEFT: 0, RIGHT: 0 };
  for (const l of lines) if (l.type === 'decision') counts[l.action.includes('LEFT') ? 'LEFT' : l.action.includes('RIGHT') ? 'RIGHT' : 'STAY']++;
  const summary = {
    score: inst.score, best: inst.best, rows_cleared: cleared, crashes: inst.crashes,
    survival_seconds: +(aliveSteps * STEP).toFixed(3), distance: +(dist + inst.dist).toFixed(2),
    decisions: decisions.length, decisions_per_game_second: +(decisions.length / SECONDS).toFixed(2),
    decisions_per_wall_second: +(decisions.length / wall).toFixed(2), wall_seconds: +wall.toFixed(2),
    max_speed: +maxSpeed.toFixed(3), speed_at_first_crash: crashes.length ? crashes[0].speed : null,
    first_crash_t_game: crashes.length ? crashes[0].t_game : null,
    choice_argmax_disagreements: lines.filter(l => l.type === 'decision' && l.model_answer !== l.argmax).length,
    action_counts: counts,
    sequence_lengths: [...new Set(lines.filter(l => l.type === 'decision').map(l => l.sequence_length))],
    routing_reasons: [...new Set(lines.filter(l => l.type === 'decision').map(l => l.routing_reason))], latency_ms_nearest_rank: { p50: q(0.5), p95: q(0.95), p99: q(0.99) },
  };
  const head = { type: 'run', run, seed, device: args.device, model: args.model, seconds: SECONDS, steps: TOTAL,
    step_s: STEP, max_rate: MAX_RATE, min_gap: MIN_GAP, params: PARAMS, started: new Date(t0wall).toISOString(), server: health };
  writeFileSync(`${args.results}${args.device}-run-${tag}.json`, JSON.stringify({ ...head, summary, crash_events: crashes, decisions,
    actions: lines.filter(l => l.type === 'decision').map(l => l.action) }));
  writeFileSync(`${args.traces}${args.device}/run-${tag}.jsonl.gz`,
    gzipSync([head, ...lines, { type: 'end', summary }].map(l => JSON.stringify(l)).join('\n') + '\n'));
  console.log(`${args.device} run ${tag} seed ${seed}`, JSON.stringify(summary));
}
