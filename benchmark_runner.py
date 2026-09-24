"""Benchmark the Lane Runner on one device: the same seeds, game and cadence for every device.

    uv run python benchmark_runner.py --device gpu
    uv run python benchmark_runner.py --device ane
    uv run python summarize.py                      # results/summary.json and results/SUMMARY.md

    # the same comparison with upstream's checkpoint for this game, kept apart from the main results
    uv run python benchmark_runner.py --device gpu --model convaiinnovations/laya --out english

Each call starts apple_server.py with only that device loaded, plays --runs games (seeds
--seed-base, --seed-base+1, ...) with tools/bench_lane_runner.mjs, and stops the server. The two
devices are never measured at the same time, and never while another laya-apple process runs:
this script refuses to start if it finds one, because a second workload on the GPU or the ANE
changes both latencies (see laya-apple research/gpu-ane-interference).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def other_laya_processes():
    """Command lines of running laya-apple or benchmark processes, except this one and its ancestors."""
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,command="], capture_output=True, text=True, env=dict(os.environ, LC_ALL="C")).stdout
    procs = {}
    for line in out.splitlines():
        pid, ppid, cmd = (line.strip().split(None, 2) + [""])[:3]
        procs[int(pid)] = (int(ppid), cmd)
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:  # the shell and uv that started us mention the same paths
        mine.add(pid)
        pid = procs[pid][0]
    keys = ("laya_apple", "apple_server.py", "interference.py", "gil_probe", "bench_concurrency", "laya-apple/.venv")
    return ["%d %s" % (p, cmd) for p, (_, cmd) in procs.items() if p not in mine and any(k in cmd for k in keys)]


def system_state():
    """Machine load at the start and end of a benchmark, recorded with the results."""
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=dict(os.environ, LC_ALL="C")).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg": list(os.getloadavg()),
            "thermal": run("pmset", "-g", "therm"), "power": run("pmset", "-g", "ps")}


def wait_ready(api, proc, name, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("apple_server.py exited with code %s before it was ready" % proc.returncode)
        try:
            with urllib.request.urlopen(api + "/api/health", timeout=2) as r:
                health = json.load(r)
            if health["models"].get(name) == "ready":
                return health
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("apple_server.py was not ready after %d s" % timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=("gpu", "ane"), required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=20260924)
    ap.add_argument("--seconds", type=int, default=90, help="game seconds per run")
    ap.add_argument("--warmup", type=int, default=50, help="untimed calls before the first run")
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--out", default="", help="subdirectory of results/ and traces/ for this set (default: none)")
    ap.add_argument("--allow-concurrent", action="store_true", help="run even if other laya-apple work is running")
    args = ap.parse_args()

    others = other_laya_processes()
    if others and not args.allow_concurrent:
        print("Other laya-apple work is running; its load would be in these numbers:\n  " + "\n  ".join(others), file=sys.stderr)
        raise SystemExit(2)

    api = "http://127.0.0.1:%d" % args.port
    seeds = [args.seed_base + i for i in range(args.runs)]
    before = system_state()
    names = {"convaiinnovations/laya": "english", "convaiinnovations/laya-multilingual": "multilingual",
             "convaiinnovations/laya-typed-decisions": "typed-decisions"}
    results, traces = os.path.join(ROOT, "results", args.out), os.path.join(ROOT, "traces", args.out)
    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "apple_server.py"), "--device", args.device, "--port", str(args.port),
                               "--model", args.model])
    try:
        health = wait_ready(api, server, names[args.model])
        print("[bench] %s ready: laya-apple %s, mlx %s, coremltools %s" % (args.device, health["laya_apple"], health["mlx"], health["coremltools"]), flush=True)
        subprocess.run(["node", os.path.join(ROOT, "tools", "bench_lane_runner.mjs"), "--api", api, "--device", args.device,
                        "--seeds", ",".join(map(str, seeds)), "--seconds", str(args.seconds), "--warmup", str(args.warmup),
                        "--model", args.model, "--results", results + os.sep, "--traces", traces + os.sep],
                       check=True)
    finally:
        server.terminate()
        server.wait(timeout=30)
    with open(os.path.join(results, "%s-system.json" % args.device), "w") as f:
        json.dump({"device": args.device, "seeds": seeds, "seconds": args.seconds, "before": before, "after": system_state(),
                   "other_laya_processes": others}, f, indent=1)


if __name__ == "__main__":
    main()
