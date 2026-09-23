# Lane Runner on laya-apple: MLX GPU vs Apple Neural Engine

The Lane Runner from [wdobry/laya-playground](https://github.com/wdobry/laya-playground), played by the same
Laya model twice: once on the Mac's GPU through MLX, once on the Apple Neural Engine through Core ML. Both
run through [laya-apple](https://github.com/tc3oliver/laya-apple). Every decision is a real model call.
Every latency comes from laya-apple's own per-call measurement.

<!-- RESULTS -->

## What is in here

This is a fork of the playground. The game (`static/demos/runner.js`), its physics, its seeded random
barriers and its decision rule are upstream's, unchanged. This fork adds:

| Path | What it is |
| --- | --- |
| `apple_server.py` | The playground server on laya-apple. `/api/predict` takes `"device": "gpu"` or `"ane"` and returns laya-apple's result, including `choice`, `probabilities`, `backend`, `device`, `latency_ms` and `routing_reason` |
| `benchmark_runner.py` | Plays the Lane Runner headlessly on one device, for a fixed set of seeds |
| `tools/bench_lane_runner.mjs` | The headless player it drives, with upstream's stepping rules from `tools/record_run.mjs` |
| `summarize.py` | Writes `results/summary.json` and `results/SUMMARY.md` from every run |
| `correctness_check.py` | Checks GPU vs ANE decisions on the game's own states, before any speed number |
| `tools/verify_replay_runs.mjs` | Checks that every recorded run replays to exactly its recorded moves and score |
| `versus.html`, `static/versus.*` | The side-by-side page: replay of recorded runs, or both devices live |
| `tools/render_video.mjs` | Renders the video from recorded runs |
| `results/` | Raw runs (`gpu-run-001.json` …), `summary.json`, `SUMMARY.md`, the correctness report |
| `traces/` | Full per-decision traces: state, barriers, model input and answer, probabilities, latency, position, score, speed, crashes |

The upstream server (`server.py`, torch and upstream `laya`) is still here and still works for the
other two games. Its README is kept as [`README.upstream.md`](README.upstream.md).

## How the comparison is kept fair

- **Same model and same prompts.** Both devices run `convaiinnovations/laya-typed-decisions` at the
  revision laya-apple pins, in FP16, answering the game's own sentence and question.
- **Same game.** Both devices play the same seeds, so they get the same barrier sequence. Initial speed,
  acceleration, the 0.25 stay threshold and the physics are upstream's.
- **Same cadence.** One request is in flight at a time, capped at 40 decisions per game second, as in the
  browser. While a request is in flight, the game keeps running: it advances by the measured round trip,
  rounded up to whole 1/120 s steps, before the answer is applied. Upstream's recorder does the same.
- **Only the device changes.** `device="gpu"` or `device="ane"`. An explicit ANE request never falls back
  to the GPU. laya-apple raises instead, and the driver also rejects any answer whose reported device is not
  the one it asked for.
- **One device at a time.** Each device is measured in its own server process, with only that device
  loaded, one after the other. `benchmark_runner.py` refuses to start while any other laya-apple process is
  running.
- **Every run counts.** Ten seeds per device, all kept. Percentiles are over every decision of every run,
  and each run's own P50, P95 and P99 are in `results/summary.json`.

The model sees only which lanes are blocked. The game's state sentence says nothing else, so every
observation is one of 7 sentences. `correctness_check.py` therefore also checks all 8 possible barrier
patterns exhaustively, alongside 120 states sampled from real games.

## Run it

You need an Apple silicon Mac, [uv](https://docs.astral.sh/uv/), Node 20 or newer, and Google Chrome (for
the video only).

```bash
git clone https://github.com/tc3oliver/laya-playground-apple
cd laya-playground-apple
uv sync                    # laya-apple[ane]==1.0.2
npm install                # playwright-core and ffmpeg-static, for the video only

# The Neural Engine needs a Core ML artifact built and parity-checked on this machine. Do this once.
uv run laya-apple artifacts build laya-typed-decisions
```

If your artifacts or weights live outside the default caches, set `LAYA_APPLE_CACHE` and `HF_HOME`.

### Correctness first

```bash
uv run python correctness_check.py      # results/CORRECTNESS.md; exits non-zero on a hard mismatch
```

### Benchmark

```bash
uv run python benchmark_runner.py --device gpu
uv run python benchmark_runner.py --device ane
uv run python summarize.py              # results/summary.json, results/SUMMARY.md
node tools/verify_replay_runs.mjs       # every run replays to its recorded moves and score
```

The defaults are 10 runs per device, seeds 20260924–20260933 and 90 game seconds per run. Run it on AC
power, on an otherwise idle machine.

### Watch it

```bash
uv run python apple_server.py           # both devices loaded
open http://127.0.0.1:8770/versus                # recorded run 1, replayed; ?run=2 … ?run=10 for the others
open "http://127.0.0.1:8770/versus?mode=live"    # both devices live, at the same time
```

In live mode, both devices share the machine at the same time. laya-apple measured that this slows both
of them down, so live numbers are not the benchmark's.

### Video

```bash
npm run render                          # video/out/lane-runner-gpu-vs-ane.mp4, 1920x1080, 30 fps
```

The video replays recorded run 1 of each device step by step: the same decisions, latencies, scores and
crashes. The video adds only three things: an 8× fast-forward in the middle (labelled on screen), the
layout, and the closing card, whose numbers are read from `results/summary.json`. To render another run,
use `RUN=3 npm run render`.

<!-- ENVIRONMENT -->

## Credits and licence

- **The Lane Runner and the playground** are by [brain function collapse](https://brainfunctioncollapse.com)
  ([wdobry/laya-playground](https://github.com/wdobry/laya-playground)), MIT licence. See [`LICENSE`](LICENSE),
  which is kept unchanged. This fork's additions are under the same licence.
- **Laya** is by [Nandakishor M](https://github.com/NandhaKishorM) (Convai Innovations), Apache-2.0:
  [code](https://github.com/NandhaKishorM/laya), [weights](https://huggingface.co/convaiinnovations).
- **laya-apple** is the Apple-native runtime: [tc3oliver/laya-apple](https://github.com/tc3oliver/laya-apple).

This fork is not affiliated with Apple, Convai Innovations or brain function collapse.
