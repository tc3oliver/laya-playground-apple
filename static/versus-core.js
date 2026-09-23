// Replay of a recorded benchmark run (results/<device>-run-NNN.json), step by step, with the
// stepping rules of the browser stage (static/live.js GameStage.tick) and upstream's
// tools/verify_replay.mjs. Browser-safe: used by the versus page, the video render and
// tools/verify_replay_runs.mjs, so all three show exactly the same game.
import runner from './demos/runner.js';
import { STEP, answersFrom } from './sim.js';

export { STEP };

export class RunReplay {
  constructor(run) {
    this.run = run; this.inst = runner.create(run.seed); this.step = 0; this.ri = 0; this.pending = null;
    this.rows = 0; this.last = 0; this.maxSpeed = this.inst.speed; this.actions = [];
    this.latest = null; this.ema = 0; this.applied = [];   // applied: steps at which decisions landed, for decisions/s
  }

  /** Advance one fixed step: apply a decision landing now, observe if one starts now, then update the game. */
  tick() {
    const { run, inst } = this, d = run.decisions;
    if (this.pending && this.ri < d.length && d[this.ri][1] === this.step) {
      const rec = d[this.ri], answers = answersFrom(this.pending, rec[2]);
      const action = inst.act(answers, run.params, true);
      this.actions.push(action.label);
      this.ema += (rec[3] - this.ema) * (this.ri ? 0.12 : 1);   // the upstream feed's smoothing (static/live.js note())
      this.latest = { i: this.ri, ms: rec[3], probs: rec[2], answer: answers.lane, action, state: this.pending.state };
      this.applied.push(this.step);
      this.pending = null; this.ri++;
    }
    if (!this.pending && this.ri < d.length && d[this.ri][0] === this.step) this.pending = inst.observe();
    const wasDead = inst.dead > 0;
    inst.update(STEP, {});
    if (inst.score > this.last) this.rows += inst.score - this.last;
    this.last = inst.score;
    if (!wasDead) this.maxSpeed = Math.max(this.maxSpeed, inst.speed);
    this.step++;
    while (this.applied.length && this.applied[0] <= this.step - 1 / STEP) this.applied.shift();
  }

  /** Step until `step` (never backwards). */
  seek(step) { while (this.step < Math.min(step, this.run.steps)) this.tick(); }

  get done() { return this.step >= this.run.steps; }
  get decisionsPerSecond() { return this.applied.length; }   // decisions landed in the last game second
}
