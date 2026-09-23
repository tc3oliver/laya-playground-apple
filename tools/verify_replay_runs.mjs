// Check that every recorded benchmark run replays to exactly what was recorded: the same action at
// every decision, and the same score, rows, crashes and top speed. Same stepping rules as upstream
// tools/verify_replay.mjs and the browser stage (static/live.js GameStage.tick).
//
//     node tools/verify_replay_runs.mjs [results-dir]
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { RunReplay } from '../static/versus-core.js';

const dir = process.argv[2] || fileURLToPath(new URL('../results/', import.meta.url));
let failed = false, n = 0;
for (const f of readdirSync(dir).filter(f => /^(gpu|ane)-run-\d+\.json$/.test(f)).sort()) {
  const run = JSON.parse(readFileSync(`${dir}/${f}`)), s = run.summary;
  const r = new RunReplay(run); r.seek(run.steps);
  const { inst, actions, rows, maxSpeed } = r;
  const ok = inst.score === s.score && inst.crashes === s.crashes && rows === s.rows_cleared && +maxSpeed.toFixed(3) === s.max_speed
    && actions.length === run.actions.length && actions.every((a, i) => a === run.actions[i]);
  failed ||= !ok; n++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${f}  ${actions.length}/${run.decisions.length} decisions, rows ${rows}/${s.rows_cleared}, crashes ${inst.crashes}/${s.crashes}`);
}
if (!n) { console.log('no runs found in ' + dir); failed = true; }
process.exit(failed ? 1 : 0);
