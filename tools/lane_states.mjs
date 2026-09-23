// Print Lane Runner observations exactly as the game sends them, for correctness_check.py.
//
//     node tools/lane_states.mjs --seeds 1,2,3 --every 3 --steps 3600 > states.json
//
// The runner is driven by an oracle pilot that reads the barrier layout directly (no model), so
// the game survives and keeps producing fresh rows. An observation is taken at the browser's
// decision cadence (every MIN_GAP steps) with the lane the runner is in when it is taken.
import { parseArgs } from 'node:util';
import runner from '../static/demos/runner.js';
import { STEP, MIN_GAP } from '../static/sim.js';

const { values: args } = parseArgs({ options: {
  seeds: { type: 'string', default: '20260924' }, every: { type: 'string', default: String(MIN_GAP) }, steps: { type: 'string', default: '3600' },
} });
const LANES = ['left', 'middle', 'right'], out = [];
for (const seed of args.seeds.split(',').map(Number)) {
  const inst = runner.create(seed);
  for (let step = 0; step < +args.steps; step++) {
    if (step % +args.every === 0 && !inst.dead) {
      const obs = inst.observe(), row = inst.nextRow();
      out.push({ seed, step, lane: LANES[inst.lane], blocked: row ? row.blocked.map(Number) : [0, 0, 0], ...obs });
      if (row && row.blocked[inst.lane]) inst.steer(Math.sign(row.blocked.indexOf(false) - inst.lane));   // oracle pilot
    }
    inst.update(STEP, {});
  }
}
process.stdout.write(JSON.stringify(out));
