"""Do the MLX GPU and the Neural Engine make the same Lane Runner decisions?

    uv run python correctness_check.py             # results/correctness.json and results/CORRECTNESS.md

The states are the game's own observations: --states of them sampled evenly from seeded games
(tools/lane_states.mjs), plus every one of the 8 possible barrier patterns, whether or not the
game produces it. Each is answered by three laya-apple instances, each in its own process, one
after the other:

  gpu      MLX, FP16 (the benchmark's GPU configuration)
  ane      Core ML on the Neural Engine, FP16, the validated artifact
  gpu32    MLX, FP32: the reference; laya-apple's parity gate holds it within 1e-4 of upstream

The comparison uses laya-apple's own FP16 gate (laya_apple.parity): a probability may move by
at most 0.02; a changed decision is a hard mismatch unless the reference's top-1/top-2 margin is
below twice that tolerance, where it is a near-tie flip and is listed, not hidden. The game's
action is also compared for every lane the runner could be in, because the game does not act on
the argmax alone: it stays while P(current lane empty) >= 0.25.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "convaiinnovations/laya-typed-decisions"
LANES = ["left", "middle", "right"]
CONFIGS = {"gpu": ("gpu", "float16"), "ane": ("ane", "float16"), "gpu32": ("gpu", "float32")}
STAY = 0.25  # static/demos/runner.js params.stay default


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def pattern_state(blocked):
    return " ".join("The %s lane is %s." % (lane, "blocked by a barrier" if b else "empty") for lane, b in zip(LANES, blocked))


def collect_states(n, seeds):
    raw = json.loads(subprocess.run(["node", os.path.join(ROOT, "tools", "lane_states.mjs"), "--seeds", ",".join(map(str, seeds))],
                                    check=True, capture_output=True, text=True).stdout)
    k = max(1, len(raw) // n)
    sampled = [dict(s, source="game") for s in raw[::k][:n]]
    questions = sampled[0]["questions"]
    patterns = [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    exhaustive = [{"source": "pattern", "blocked": p, "lane": None, "state": pattern_state(p), "questions": questions} for p in patterns]
    for s in sampled:  # the game's text and ours must agree, or the patterns test something else
        assert s["state"] == pattern_state(s["blocked"]), s
    return sampled + exhaustive


def worker(config, states_path, out_path):
    from laya_apple import Laya

    device, dtype = CONFIGS[config]
    states = read_json(states_path)
    laya = Laya.from_pretrained(MODEL, device=device, dtype=dtype)
    for _ in range(20):
        laya.predict(context=states[0]["state"], questions=states[0]["questions"])
    rows = []
    for s in states:
        r = laya.predict(context=s["state"], questions=s["questions"])
        again = laya.predict(context=s["state"], questions=s["questions"])
        a, b = r.answers["lane"], again.answers["lane"]
        rows.append({"choice": a["choice"], "probabilities": [a["probabilities"][x] for x in LANES],
                     "repeat_identical": a["probabilities"] == b["probabilities"], "latency_ms": r.runtime.latency_ms,
                     "backend": r.runtime.backend, "device": r.runtime.device, "routing_reason": r.runtime.routing_reason,
                     "sequence_length": r.runtime.sequence_length, "artifact_revision": r.runtime.artifact_revision,
                     "dtype": r.runtime.dtype, "buckets": list(r.runtime.buckets)})
    write(out_path, json.dumps({"config": config, "device": device, "dtype": dtype, "info": laya.info(), "rows": rows}))


def game_action(probs, lane):
    """static/demos/runner.js act(): stay while P(current lane empty) >= STAY, else step towards the argmax."""
    cur = LANES.index(lane)
    if probs[cur] >= STAY:
        return "STAY"
    want = max(range(3), key=lambda i: probs[i])  # answersFrom: first index of the maximum
    return "LEFT" if want < cur else "RIGHT" if want > cur else "STAY"


def compare(states, out):
    from laya_apple.parity import TOLERANCE

    tol = TOLERANCE["float16"]
    ref = out["gpu32"]["rows"]
    report = {"model": MODEL, "tolerance": tol, "near_tie_band": 2 * tol, "stay_threshold": STAY, "states": len(states),
              "states_from_games": sum(s["source"] == "game" for s in states),
              "unique_prompts": len({s["state"] for s in states}), "configs": {}, "pairs": {}}
    for c, o in out.items():
        report["configs"][c] = {"device": o["device"], "dtype": o["dtype"], "sequence_lengths": sorted({r["sequence_length"] for r in o["rows"]}),
                                "routing_reasons": sorted({r["routing_reason"] for r in o["rows"]}),
                                "artifact_revisions": sorted({r["artifact_revision"] for r in o["rows"]}),
                                "all_repeats_identical": all(r["repeat_identical"] for r in o["rows"]),
                                "ane_buckets": o["info"].get("ane_buckets")}
    for a, b in (("gpu", "ane"), ("gpu", "gpu32"), ("ane", "gpu32")):
        ra, rb = out[a]["rows"], out[b]["rows"]
        diffs, hard, ties, act_mismatch = [], [], [], []
        for i, s in enumerate(states):
            d = max(abs(x - y) for x, y in zip(ra[i]["probabilities"], rb[i]["probabilities"]))
            diffs.append(d)
            srt = sorted(ref[i]["probabilities"])
            margin = srt[-1] - srt[-2]
            if ra[i]["choice"] != rb[i]["choice"]:
                (ties if margin < 2 * tol else hard).append({"i": i, "state": s["state"], a: ra[i]["choice"], b: rb[i]["choice"], "ref_margin": margin})
            for lane in ([s["lane"]] if s["lane"] else LANES):
                x, y = game_action(ra[i]["probabilities"], lane), game_action(rb[i]["probabilities"], lane)
                if x != y:
                    act_mismatch.append({"i": i, "state": s["state"], "lane": lane, a: x, b: y})
        report["pairs"]["%s_vs_%s" % (a, b)] = {
            "prob_max_abs": max(diffs), "within_tolerance": max(diffs) <= tol, "hard_mismatches": hard,
            "near_tie_flips": ties, "game_action_mismatches": act_mismatch,
            "chosen_action_agreement": sum(ra[i]["choice"] == rb[i]["choice"] for i in range(len(states))) / len(states)}
    uniq = {}
    for i, s in enumerate(states):
        uniq.setdefault(s["state"], {"state": s["state"], "count": 0, **{c: {"choice": out[c]["rows"][i]["choice"],
                                                                           "probabilities": [round(p, 6) for p in out[c]["rows"][i]["probabilities"]]}
                                                                        for c in out}})
        uniq[s["state"]]["count"] += 1
    report["by_prompt"] = list(uniq.values())
    return report


def md(r):
    L = ["# Lane Runner correctness: MLX GPU vs Apple Neural Engine", "", "Generated by `correctness_check.py`. Do not edit by hand.", "",
         "%d states: %d sampled from seeded games, plus all 8 barrier patterns. They contain %d distinct prompts: the game's state "
         "sentence only says which lanes are blocked, so every observation is one of these." % (r["states"], r["states_from_games"], r["unique_prompts"]), "",
         "Gate: laya-apple's FP16 parity gate. Probability tolerance %.2f; a changed decision is a hard mismatch unless the FP32 reference's "
         "top-1/top-2 margin is below %.2f (a near-tie flip, listed)." % (r["tolerance"], r["near_tie_band"]), "",
         "| Pair | Max prob. difference | Within tolerance | Same chosen lane | Hard mismatches | Near-tie flips | Game-action mismatches |",
         "|---|---:|---|---:|---:|---:|---:|"]
    for k, p in r["pairs"].items():
        L.append("| %s | %.5f | %s | %.1f%% | %d | %d | %d |" % (k.replace("_vs_", " vs "), p["prob_max_abs"], "yes" if p["within_tolerance"] else "**no**",
                                                           100 * p["chosen_action_agreement"], len(p["hard_mismatches"]), len(p["near_tie_flips"]), len(p["game_action_mismatches"])))
    L += ["", "| Config | Device | Precision | Prompt tokens | Routing | Repeat calls identical |", "|---|---|---|---|---|---|"]
    for c, x in r["configs"].items():
        L.append("| %s | %s | %s | %s | %s | %s |" % (c, x["device"], x["dtype"], x["sequence_lengths"], ", ".join(x["routing_reasons"]), x["all_repeats_identical"]))
    L += ["", "## Every distinct prompt", "", "| Prompt | Seen | GPU FP16 | ANE FP16 | GPU FP32 (ref.) |", "|---|---:|---|---|---|"]
    def cell(x):
        return "%s (%s)" % (x["choice"], " / ".join("%.3f" % p for p in x["probabilities"]))

    for u in r["by_prompt"]:
        L.append("| %s | %d | %s | %s | %s |" % (u["state"], u["count"], cell(u["gpu"]), cell(u["ane"]), cell(u["gpu32"])))
    L += ["", "Probabilities are listed left / middle / right.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--seeds", default="20260924,20260925,20260926")
    ap.add_argument("--worker", choices=CONFIGS)
    ap.add_argument("--states-file")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker, args.states_file, args.out)

    res = os.path.join(ROOT, "results")
    os.makedirs(res, exist_ok=True)
    states = collect_states(args.states, [int(x) for x in args.seeds.split(",")])
    sp = os.path.join(res, "correctness-states.json")
    write(sp, json.dumps(states))
    out = {}
    for c in CONFIGS:  # one process per configuration, never two at once
        op = os.path.join(res, "correctness-%s.json" % c)
        subprocess.run([sys.executable, __file__, "--worker", c, "--states-file", sp, "--out", op], check=True)
        out[c] = read_json(op)
    report = compare(states, out)
    write(os.path.join(res, "correctness.json"), json.dumps(report, indent=1))
    write(os.path.join(res, "CORRECTNESS.md"), md(report))
    print(md(report))
    hard = sum(len(p["hard_mismatches"]) for p in report["pairs"].values())
    return 1 if hard or not report["pairs"]["gpu_vs_ane"]["within_tolerance"] else 0


if __name__ == "__main__":
    sys.exit(main())
