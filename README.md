# Lane Runner on laya-apple: MLX GPU vs Apple Neural Engine

The Lane Runner from [wdobry/laya-playground](https://github.com/wdobry/laya-playground), played by the same
Laya model twice: once on the Mac's GPU through MLX, once on the Apple Neural Engine through Core ML. Both
run through [laya-apple](https://github.com/tc3oliver/laya-apple). Every decision is a real model call.
Every latency comes from laya-apple's own per-call measurement.

## Result

Measured on an Apple M4 Max: 10 runs per device, the same 10 seeds, 90 game seconds each, 28,826
decisions per device. Source: [`results/SUMMARY.md`](results/SUMMARY.md) and
[`results/summary.json`](results/summary.json), generated from the raw runs.

| | MLX GPU | Apple Neural Engine |
| --- | ---: | ---: |
| Latency P50 | 9.24 ms | 8.19 ms |
| Latency P95 | 9.69 ms | 8.30 ms |
| Latency P99 | 9.76 ms | 8.40 ms |
| Latency mean | 9.33 ms | 8.20 ms |
| Latency max | 10.75 ms | 27.01 ms |
| Rows cleared, median run (min–max) | 51.5 (32–57) | 51.5 (32–57) |
| Crashes, all runs | 164 | 164 |
| Top game speed | 23.18 | 23.18 |
| Decisions per game second, median | 32.43 | 32.43 |

- **Both devices make the same decisions.** Across 128 game states, GPU and ANE probabilities differ
  by at most 0.0043, against laya-apple's FP16 tolerance of 0.02. There are no hard mismatches and no
  near-tie flips, and the game's move is the same in every case
  ([`results/CORRECTNESS.md`](results/CORRECTNESS.md)).
- **The ANE answers about 1 ms sooner:** P50 8.19 ms against 9.24 ms, P95 8.30 ms against 9.69 ms. The
  ANE had two slow calls out of 28,826 (25.2 and 27.0 ms); its P99 is 8.40 ms. The GPU's slowest was 10.75 ms.
- **The game result is the same on both devices, seed for seed.** Rows cleared, crashes, score, top speed
  and action counts match on all 10 seeds. At this game's cap of 40 decisions per second, both devices
  keep up. The ANE's shorter round trip sometimes lands an answer one 1/120 s step sooner, so on one
  seed two moves come in a different order, with the same outcome.
- **The runner crashes often on both devices, for the same reason.** Take the middle lane with the left
  and middle blocked. `laya-typed-decisions` gives P(middle empty) = 0.26 there, and the game stays
  whenever P(current lane empty) ≥ 0.25, upstream's default. The runner stays and hits the barrier. Both
  devices and the FP32 reference agree, so it is how this checkpoint answers this prompt, not a device
  difference. Every crash resets the speed, which is why the top speed stays near 23 of the game's 31.
  Upstream plays this game with the `english` checkpoint instead.


### The same comparison with upstream's checkpoint

Upstream plays the Lane Runner with the `english` checkpoint (`convaiinnovations/laya`). The same pipeline,
the same 10 seeds, with results kept apart in [`results/english/`](results/english/SUMMARY.md):

| `convaiinnovations/laya` | MLX GPU | Apple Neural Engine |
| --- | ---: | ---: |
| Latency P50 | 9.24 ms | 8.19 ms |
| Latency P95 | 9.69 ms | 8.30 ms |
| Latency P99 | 9.76 ms | 8.40 ms |
| Latency max | 13.62 ms | 29.16 ms |
| Rows cleared, every run | 185 | 185 |
| Crashes, all runs | 0 | 0 |
| Top game speed | 31.0 (the game's cap) | 31.0 (the game's cap) |
| Decisions per game second | 40.0 (the cap) | 40.0 (the cap) |

- GPU and ANE probabilities differ by at most 0.0020, with 0 mismatches of any kind
  ([`results/english/CORRECTNESS.md`](results/english/CORRECTNESS.md)).
- Every blocked lane gets P ≤ 0.10, so the runner never stays in one. It reaches the top speed and never
  crashes on either device.
- With no crash, the score does not depend on the seed. The row spacing and speed depend only on score
  and time, so every run clears 185 rows, although the barrier patterns differ from seed to seed.
- The latencies match the `typed-decisions` runs to within 0.01 ms. Both checkpoints have the same
  architecture and produce the same prompt lengths (36–45 tokens), so they cost the same.

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

# upstream's checkpoint for this game, into results/english/ and traces/english/
uv run python correctness_check.py --model convaiinnovations/laya --out results/english
uv run python benchmark_runner.py --device gpu --model convaiinnovations/laya --out english
uv run python benchmark_runner.py --device ane --model convaiinnovations/laya --out english
uv run python summarize.py --results results/english
node tools/verify_replay_runs.mjs results/english
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
SET=english npm run render              # video/out/lane-runner-gpu-vs-ane-english.mp4
```

The video replays recorded run 1 of each device step by step: the same decisions, latencies, scores and
crashes. The video adds only three things: an 8× fast-forward in the middle (labelled on screen), the
layout, and the closing card, whose numbers are read from `results/summary.json`. To render another run,
use `RUN=3 npm run render`.

## Environment of the recorded results

| | |
| --- | --- |
| Hardware | Apple M4 Max, 64 GB, on AC power |
| macOS | 26.6.2 (25G83) |
| Python | 3.12.14 (uv) |
| laya-apple | 1.0.2 from PyPI (tag `v1.0.2`, `d874805`) |
| MLX | 0.32.2 |
| coremltools | 9.0 |
| Model | `convaiinnovations/laya-typed-decisions`, revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, FP16 on both devices |
| ANE artifact (english) | `convaiinnovations/laya` revision `c5d78730f3493e4fe16d61507ef4b78eef7318cf`, `bc1s-masked` L64, `artifact_sha256` `f6263eb65898…`, 10,594 of 10,594 ops on the Neural Engine, parity passed (probability max \|Δ\| 0.0093), built locally |
| ANE artifact (typed-decisions) | `bc1s-masked`, bucket L64 (every prompt is 36–45 tokens), `artifact_sha256` `1273fcd30495…`, 10,594 ops of 10,594 on the Neural Engine with 0 transitions, parity passed (probability max \|Δ\| 0.0046, 0 hard mismatches), built locally by `laya-apple artifacts build` |
| GPU runtime | MLX, weights `mlx:4fa56de72383`, FP16 |
| Load | Load average and `pmset` thermal state before and after each device are in `results/<device>-system.json`. No other laya-apple process was running |


## Credits and licence

- **The Lane Runner and the playground** are by [brain function collapse](https://brainfunctioncollapse.com)
  ([wdobry/laya-playground](https://github.com/wdobry/laya-playground)), MIT licence. See [`LICENSE`](LICENSE),
  which is kept unchanged. This fork's additions are under the same licence.
- **Laya** is by [Nandakishor M](https://github.com/NandhaKishorM) (Convai Innovations), Apache-2.0:
  [code](https://github.com/NandhaKishorM/laya), [weights](https://huggingface.co/convaiinnovations).
- **laya-apple** is the Apple-native runtime: [tc3oliver/laya-apple](https://github.com/tc3oliver/laya-apple).

This fork is not affiliated with Apple, Convai Innovations or brain function collapse.
