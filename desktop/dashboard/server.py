"""Tableau de bord local du pipeline NIDS (bibliothèque standard uniquement).

Lancement, depuis la racine du projet :
    python -m dashboard.server            puis ouvrir http://127.0.0.1:8765

Le serveur n'écoute que sur 127.0.0.1. Les actions (POST) exigent un jeton généré à chaque démarrage et un en-tête Host
local, ce qui empêche une page web tierce de lancer des étapes depuis votre navigateur.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dashboard.pipeline import Config, JobStore, Runner, build_steps, compute_states, validate_config

HERE = Path(__file__).resolve().parent
ALLOWED_EXT = {".md", ".png", ".jpg", ".jpeg", ".csv", ".txt", ".json"}
MAX_REPORT_CHARS = 60000


class App:
    def __init__(self, root: Path, port: int):
        self.root, self.port = root, port
        self.token = secrets.token_urlsafe(24)
        self.cfg_path = HERE / "config.json"
        self.store = JobStore(HERE / "state.json")
        self.runner = Runner(root, self.store, HERE / "logs")
        self.cfg = self._load_cfg()
        self.lock = threading.Lock()

    def _load_cfg(self) -> Config:
        d = {}
        if self.cfg_path.exists():
            try:
                d = json.loads(self.cfg_path.read_text(encoding="utf-8"))
            except Exception:
                d = {}
        try:
            return validate_config(d, self.root)
        except ValueError:
            return validate_config({}, self.root)

    def set_cfg(self, d: dict) -> Config:
        cfg = validate_config(d, self.root)
        with self.lock:
            self.cfg = cfg
            self.cfg_path.write_text(json.dumps(cfg.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        return cfg

    def steps(self):
        return build_steps(self.cfg)

    def overview(self) -> dict:
        steps = self.steps()
        st = compute_states(self.cfg, steps, self.store)
        mand = [s for s in st if not s["optional"]]
        job = None
        if self.runner.busy or self.runner.step_id:
            sid = self.runner.step_id
            last = self.store.last(sid) if sid else None
            if last:
                job = {"step": sid, "title": next((s["title"] for s in st if s["id"] == sid), ""), "running": self.runner.busy,
                       "status": last.get("status"), "started": last.get("started"), "ended": last.get("ended"),
                       "log": self.runner.tail(last.get("log", "")), "log_name": last.get("log")}
        return {"config": self.cfg.to_json(), "steps": st, "groups": sorted({s["group"] for s in st}, key=lambda g: [x["group"] for x in st].index(g)),
                "progress": {"done": sum(1 for s in mand if s["status"] in ("termine", "a_refaire")), "total": len(mand)},
                "job": job, "root": str(self.root)}

    def report(self, step_id: int) -> dict:
        steps = {s.id: s for s in self.steps()}
        s = steps.get(step_id)
        if not s:
            raise KeyError(step_id)
        docs = []
        for rel in s.reports:
            p = (self.root / rel).resolve()
            if self.root.resolve() in p.parents and p.exists() and p.suffix in ALLOWED_EXT:
                txt = p.read_text(encoding="utf-8", errors="replace")
                docs.append({"name": rel, "text": txt[:MAX_REPORT_CHARS], "truncated": len(txt) > MAX_REPORT_CHARS})
        imgs = []
        for u in s.units:
            for it in u:
                if isinstance(it, str) and it.endswith(".png") and (self.root / it).exists():
                    imgs.append(it)
        if step_id in (19, 28):
            for p in sorted((self.root / "reports/figures").glob(f"*_s{self.cfg.head_seed}*.png")):
                rel = p.relative_to(self.root).as_posix()
                if rel not in imgs:
                    imgs.append(rel)
        return {"docs": docs, "images": imgs}

    def run(self, step_id: int):
        steps = {s.id: s for s in self.steps()}
        s = steps.get(step_id)
        if not s:
            raise KeyError(step_id)
        st = {x["id"]: x for x in compute_states(self.cfg, list(steps.values()), self.store)}[step_id]
        if st["status"] == "verrouille":
            raise RuntimeError("Étape verrouillée : " + (st["blocked_reason"] or "terminez d'abord " + ", ".join(st["waiting_for"])))
        if st["status"] == "en_cours":
            raise RuntimeError("Cette étape est déjà en cours.")
        return self.runner.start(s, self.cfg.python)


def make_handler(app: App):
    host_ok = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}
    origin_ok = {f"http://{h}" for h in host_ok}

    class H(BaseHTTPRequestHandler):
        server_version = "NIDSDash/1"

        def log_message(self, *a):   # silencieux
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _guard(self) -> bool:
            if self.headers.get("Host", "") not in host_ok:
                self._send(403, {"error": "hôte refusé"})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    html = (HERE / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.token)
                    return self._send(200, html, "text/html; charset=utf-8")
                if u.path == "/api/overview":
                    return self._send(200, app.overview())
                if u.path == "/api/report":
                    q = parse_qs(u.query)
                    return self._send(200, app.report(int(q.get("id", ["0"])[0])))
                if u.path == "/api/log":
                    q = parse_qs(u.query)
                    name = Path(q.get("name", [""])[0]).name
                    return self._send(200, {"text": app.runner.tail(name, 400)})
                if u.path.startswith("/files/"):
                    rel = Path(u.path[len("/files/"):])
                    p = (app.root / rel).resolve()
                    base = (app.root / "reports").resolve()
                    if base in p.parents and p.is_file() and p.suffix in ALLOWED_EXT:
                        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                        return self._send(200, p.read_bytes(), ctype)
                    return self._send(404, {"error": "introuvable"})
                return self._send(404, {"error": "introuvable"})
            except (KeyError, ValueError):
                return self._send(404, {"error": "introuvable"})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            if not self._guard():
                return
            if self.headers.get("X-Token") != app.token:
                return self._send(403, {"error": "jeton invalide : rechargez la page"})
            origin = self.headers.get("Origin")
            if origin and origin not in origin_ok:
                return self._send(403, {"error": "origine refusée"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 20000:
                    return self._send(413, {"error": "requête trop grande"})
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/config":
                    app.set_cfg(body)
                    return self._send(200, app.overview())
                if path == "/api/run":
                    log = app.run(int(body["id"]))
                    return self._send(200, {"ok": True, "log": log})
                if path == "/api/stop":
                    app.runner.stop()
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "introuvable"})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except (KeyError, TypeError):
                return self._send(400, {"error": "requête invalide"})
            except RuntimeError as e:
                return self._send(409, {"error": str(e)})
            except Exception as e:
                return self._send(500, {"error": str(e)})

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tableau de bord local du pipeline NIDS")
    ap.add_argument("--root", default=".", help="racine du projet (dossier contenant src, scripts, data, reports)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if not (root / "src").is_dir():
        print(f"Dossier « src » introuvable dans {root} : lancez la commande depuis la racine du projet.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))
    (root / "reports").mkdir(exist_ok=True)
    app = App(root, a.port)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    url = f"http://127.0.0.1:{a.port}"
    print(f"Tableau de bord : {url}   (Ctrl+C pour arrêter)\nRacine du projet : {root}")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
        app.runner.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
