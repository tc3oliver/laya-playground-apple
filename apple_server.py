"""The playground server on laya-apple: the same pages and JSON API as server.py, but every
decision runs on laya-apple, on a device the request names.

    uv run python apple_server.py                    # MLX GPU and the Neural Engine, both loaded
    uv run python apple_server.py --device gpu       # one device only: what the benchmark uses
    uv run python apple_server.py --device ane

Then open http://127.0.0.1:8770/versus for the side-by-side Lane Runner.

POST /api/predict takes the upstream body ({state, questions, model}) plus "device": "gpu" |
"ane". Nothing is mocked, cached or replayed: every call is one laya-apple forward pass, and
the latency reported is laya-apple's own RuntimeInfo.latency_ms for that call.

Standard library plus laya-apple. Binds to loopback; nothing is reachable from the network.
"""

import argparse
import json
import mimetypes
import os
import platform
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import laya_apple
from laya_apple import InvalidRequestError, Laya, LayaAppleError

HOST, ROOT = "127.0.0.1", os.path.dirname(os.path.abspath(__file__))
# Copied from server.py, which cannot be imported here: it loads torch and upstream laya at import.
PAGES = {"/": "index.html", "/index.html": "index.html", "/about": "about.html", "/about.html": "about.html",
         "/playground": "playground.html", "/playground.html": "playground.html",
         "/robots.txt": "robots.txt", "/sitemap.xml": "sitemap.xml", "/og.png": "og.png", "/favicon.svg": "favicon.svg",
         "/versus": "versus.html", "/versus.html": "versus.html"}
PUBLIC = tuple(os.path.realpath(os.path.join(ROOT, d)) + os.sep for d in ("static", "skills", "results"))
DEVICES = ("gpu", "ane")
DEFAULT_MODEL = "convaiinnovations/laya-typed-decisions"
# The upstream page names its checkpoints; laya-apple names the same Hugging Face repositories.
CHECKPOINTS = {"english": "convaiinnovations/laya", "multilingual": "convaiinnovations/laya-multilingual",
               "typed-decisions": "convaiinnovations/laya-typed-decisions"}
MAX_BODY = 1 << 20


class Runtimes:
    """One laya-apple instance per (checkpoint, device), each behind its own lock.

    The GPU and the ANE instances are separate objects, so a GPU call never waits for an ANE
    call. Calls to the same instance are serialised: laya-apple's inline execution runs one
    request at a time, and the benchmark measures exactly that.
    """

    def __init__(self, devices):
        self.devices = devices
        self.inst, self.locks, self.status = {}, {}, {}
        self.guard = threading.Lock()

    def key(self, model, device):
        repo = CHECKPOINTS.get(model or "typed-decisions", model)
        if device not in self.devices:
            raise ValueError("device %r is not loaded by this server (loaded: %s)" % (device, ", ".join(self.devices)))
        return repo, device

    def get(self, repo, device):
        with self.guard:
            lock = self.locks.setdefault((repo, device), threading.Lock())
        with lock:
            if (repo, device) not in self.inst:
                self.status[(repo, device)] = "loading"
                try:
                    self.inst[(repo, device)] = Laya.from_pretrained(repo, device=device)
                except Exception as e:
                    self.status[(repo, device)] = "error: %s" % e
                    raise
                self.status[(repo, device)] = "ready"
        return self.inst[(repo, device)], lock

    def predict(self, state, questions, model, device):
        repo, device = self.key(model, device)
        laya, lock = self.get(repo, device)
        with lock:
            result = laya.predict(context=state, questions=questions)
        return result

    def health(self):
        # Upstream's pages read {"models": {checkpoint: status}} and play live once theirs is "ready".
        by_name = {name: "pending" for name in CHECKPOINTS}
        for name, repo in CHECKPOINTS.items():
            states = [self.status.get((repo, d)) for d in self.devices]
            if all(s == "ready" for s in states):
                by_name[name] = "ready"
            elif any(s for s in states):
                by_name[name] = next(s for s in states if s and s != "ready")
        return by_name


RUNTIMES: Runtimes = None  # set in main()


def environment():
    """What produced a number: recorded next to every result."""
    import coremltools
    import mlx.core as mx

    soc = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {
        "soc": soc or None,
        "laya_apple": laya_apple.__version__,
        "mlx": mx.__version__,
        "coremltools": coremltools.__version__,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
    }


def validate(questions):
    """server.py's check, unchanged, so a bad request fails the same way on both servers."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of id -> definition")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            raise ValueError("question %r: type must be choice, score or noul" % qid)
        if not q.get("instructions"):
            raise ValueError("question %r: instructions are required" % qid)
        crit = q.get("criteria")
        if q["type"] == "choice" and not (isinstance(crit, (dict, list)) and len(crit) >= 2):
            raise ValueError("question %r: a choice needs at least 2 options" % qid)
        if q["type"] == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise ValueError("question %r: a score needs at least 2 ordered levels" % qid)


def predict(payload):
    state, questions = payload.get("state"), payload.get("questions")
    if state in (None, "", {}, []):
        raise ValueError("state is empty")
    validate(questions)
    device = payload.get("device") or ("gpu" if "gpu" in RUNTIMES.devices else RUNTIMES.devices[0])
    t0 = time.perf_counter()
    result = RUNTIMES.predict(state, questions, payload.get("model"), device)
    server_ms = (time.perf_counter() - t0) * 1000
    out = result.to_dict()
    rt = result.runtime
    first = next(iter(result.answers.values()))
    out.update({
        # the flat fields the Lane Runner overlay reads; all of them come from laya-apple's result
        "choice": first.get("choice"),
        "probabilities": first.get("probabilities", {}),
        "backend": rt.backend,
        "device": rt.device,
        "latency_ms": rt.latency_ms,
        "routing_reason": rt.routing_reason,
        "sequence_length": rt.sequence_length,
        "artifact_revision": rt.artifact_revision,
        "server_ms": round(server_ms, 3),  # laya-apple latency plus the lock wait; for diagnosis only
    })
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "laya-playground-apple"
    protocol_version = "HTTP/1.1"  # keep-alive: the demos send 40 requests a second per device

    def log_request(self, code="-", size="-"):
        if str(code) == "200" and self.path.startswith(("/api/health", "/api/predict", "/static/", "/results/")):
            return
        super().log_request(code, size)

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):  # DNS-rebinding guard, as upstream
        host = (self.headers.get("Host") or "").split(":")[0]
        if host in ("127.0.0.1", "localhost"):
            return True
        self._send(403, {"error": "forbidden host"})
        return False

    def do_GET(self):
        if not self._host_ok():
            return
        path = self.path.split("?")[0]
        if path in PAGES:
            self._static(PAGES[path], root_file=True)
        elif path == "/api/health":
            self._send(200, {"models": RUNTIMES.health(), "devices": list(RUNTIMES.devices),
                             "version": "laya-apple " + laya_apple.__version__, **environment()})
        elif path.startswith(("/static/", "/skills/", "/results/")):
            self._static(path.lstrip("/"))
        else:
            self._send(404, {"error": "not found"})

    def _static(self, rel, root_file=False):
        full = os.path.realpath(os.path.join(ROOT, rel))
        if not (root_file or full.startswith(PUBLIC)) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = {".js": "text/javascript", ".md": "text/markdown", ".html": "text/html", ".svg": "image/svg+xml",
                 ".json": "application/json"}.get(os.path.splitext(full)[1]) or mimetypes.guess_type(full)[0] \
            or "application/octet-stream"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        if not self._host_ok():
            return
        if self.path != "/api/predict":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ValueError("request body must be between 1 byte and 1 MB")
            self._send(200, predict(json.loads(self.rfile.read(n))))
        except (ValueError, KeyError, TypeError, InvalidRequestError) as e:
            self._send(400, {"error": str(e)})
        except LayaAppleError as e:  # e.g. an explicit ANE request with no validated artifact: never rerouted
            self._send(503, {"error": "%s: %s" % (type(e).__name__, e)})
        except Exception as e:
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    global RUNTIMES
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=DEVICES + ("both",), default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint loaded at start (others load on first use)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8770")))
    args = ap.parse_args()
    RUNTIMES = Runtimes(DEVICES if args.device == "both" else (args.device,))
    for device in RUNTIMES.devices:  # load before serving: a benchmark never measures a cold load
        t0 = time.perf_counter()
        RUNTIMES.get(args.model, device)
        print("[laya-apple] %s on %s ready in %.1fs" % (args.model, device, time.perf_counter() - t0), flush=True)
    print("laya-apple %s serving %s\n  versus   http://%s:%d/versus\n  landing  http://%s:%d/"
          % (laya_apple.__version__, "+".join(RUNTIMES.devices), HOST, args.port, HOST, args.port), flush=True)
    ThreadingHTTPServer((HOST, args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
