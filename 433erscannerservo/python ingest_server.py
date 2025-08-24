#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
433 MHz Ingest + Phone Viewer (Session-Modus)
- /ingest  : ESP32 postet CSV-Zeilen (text/csv)
- /        : Handy-Viewer zeigt immer das neueste Dashboard-PNG
- /latest.png : liefert dashboard_latest.png oder das neueste dashboard_live_*.png
- /reset   : löscht generierte Dateien und startet eine frische CSV-Session

Session-Features:
- --fresh-start     : löscht alte CSV/PNG beim Start und erzeugt neue Session-CSV
- --purge-on-exit   : löscht generierte Dateien bei sauberem Exit (Default an)

Es werden nur folgende Muster im captures-Ordner gelöscht:
  rx_*.csv, dashboard_live_*.png, dashboard_latest.png, signatures_*.csv
Hintergrundbilder/sonstige Assets bleiben erhalten.
"""

import os, glob, time, atexit, signal, argparse
from datetime import datetime
from flask import Flask, request, send_file, abort, render_template_string, jsonify

app = Flask(__name__)

# Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR  = os.path.join(BASE_DIR, "captures")
os.makedirs(CAP_DIR, exist_ok=True)

# Globale Session-Variablen
ACTIVE_CSV = None
CFG = {
    "purge_on_exit": True,
    "fresh_start": False,
    "viewer_interval": 2.0,   # Sekunden Auto-Refresh im Handy-Viewer
}

HTML = """
<!DOCTYPE html><html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>433 MHz Live</title>
<style>
  html,body{margin:0;background:#0b0f13;color:#bfe;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
  header{position:fixed;top:0;left:0;right:0;padding:10px 12px;background:rgba(11,15,19,.6);
         backdrop-filter:blur(4px);font-weight:600}
  .wrap{padding-top:52px} img{width:100%;height:auto;display:block}
  .muted{color:#8aa;font-size:12px}
</style></head><body>
<header>433 MHz – Dashboard (LIVE) <span class="muted">| Auto-Refresh {{interval}}s</span></header>
<div class="wrap"><img id="dash" alt="dashboard" src="/latest.png?ts={{ts}}"></div>
<script>
const img=document.getElementById('dash');const ms={{interval}}*1000;
setInterval(()=>{img.src="/latest.png?ts="+Date.now()},ms);
</script></body></html>
"""

# ---------------------- Datei-Helfer ----------------------
def _session_csv_name() -> str:
    ts = datetime.now().strftime("rx_session_%Y%m%d_%H%M%S.csv")
    return os.path.join(CAP_DIR, ts)

def _purge_generated() -> int:
    """Nur generierte Dateien löschen (CSV/PNGs/Signatur-Exports)."""
    patterns = [
        os.path.join(CAP_DIR, "rx_*.csv"),
        os.path.join(CAP_DIR, "dashboard_live_*.png"),
        os.path.join(CAP_DIR, "dashboard_latest.png"),
        os.path.join(CAP_DIR, "signatures_*.csv"),
    ]
    removed = 0
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
                removed += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                print("⚠️  Konnte nicht löschen:", p, e)
    return removed

def _ensure_latest_png() -> str | None:
    """Bevorzugt dashboard_latest.png, sonst neuestes dashboard_live_*.png."""
    latest = os.path.join(CAP_DIR, "dashboard_latest.png")
    if os.path.exists(latest):
        return latest
    lives = glob.glob(os.path.join(CAP_DIR, "dashboard_live_*.png"))
    if lives:
        return max(lives, key=os.path.getmtime)
    return None

def _start_new_session():
    """Neue Session-CSV anlegen und global setzen."""
    global ACTIVE_CSV
    ACTIVE_CSV = _session_csv_name()
    # Leere Datei erzeugen (ohne Header; wir speichern Rohzeilen)
    with open(ACTIVE_CSV, "w", encoding="utf-8") as f:
        pass
    print("🆕 Neue Session gestartet:", os.path.basename(ACTIVE_CSV))

# ---------------------- Flask Routes ----------------------
@app.route("/ingest", methods=["POST"])
def ingest():
    global ACTIVE_CSV
    if ACTIVE_CSV is None:
        _start_new_session()

    data = request.get_data(cache=False, as_text=True)
    if not data:
        return jsonify({"ok": False, "err": "empty body"}), 400

    try:
        with open(ACTIVE_CSV, "a", encoding="utf-8") as f:
            f.write(data)
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500

    return jsonify({"ok": True, "written": len(data), "file": os.path.basename(ACTIVE_CSV)})

@app.route("/")
def index():
    return render_template_string(HTML, interval=CFG["viewer_interval"], ts=int(time.time()))

@app.route("/latest.png")
def latest_png():
    path = _ensure_latest_png()
    if not path:
        abort(404, "Kein Dashboard-PNG gefunden. Läuft analyze_signals.py und speichert PNGs?")
    return send_file(path, mimetype="image/png", as_attachment=False, conditional=False)

@app.route("/reset", methods=["POST", "GET"])
def reset():
    """Zur Laufzeit alles bereinigen und neue Session starten."""
    removed = _purge_generated()
    _start_new_session()
    return jsonify({"ok": True, "removed": removed, "active_csv": os.path.basename(ACTIVE_CSV)})

@app.route("/status")
def status():
    latest = _ensure_latest_png()
    return jsonify({
        "active_csv": os.path.basename(ACTIVE_CSV) if ACTIVE_CSV else None,
        "has_png": bool(latest),
        "png_file": os.path.basename(latest) if latest else None,
    })

# ---------------------- Lifecycle / CLI ----------------------
def _on_exit(*_):
    if CFG["purge_on_exit"]:
        removed = _purge_generated()
        print(f"🧹 Aufräumen bei Exit: {removed} Dateien gelöscht.")

def main():
    parser = argparse.ArgumentParser(description="433 MHz Ingest + Phone Viewer (Session-Modus)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP-Port (Default 8000)")
    parser.add_argument("--fresh-start", action="store_true",
                        help="Beim Start alle generierten Dateien löschen und neue CSV beginnen")
    parser.add_argument("--no-purge-on-exit", action="store_true",
                        help="Keine automatische Bereinigung beim Beenden")
    parser.add_argument("--viewer-interval", type=float, default=2.0,
                        help="Auto-Refresh-Intervall im Handy-Viewer (Sekunden)")
    args = parser.parse_args()

    CFG["fresh_start"] = args.fresh_start
    CFG["purge_on_exit"] = not args.no_purge_on_exit
    CFG["viewer_interval"] = float(args.viewer_interval)

    if CFG["fresh_start"]:
        removed = _purge_generated()
        print(f"🧼 Fresh-Start: {removed} alte Dateien gelöscht.")
    _start_new_session()

    # Cleanup registrieren
    atexit.register(_on_exit)
    signal.signal(signal.SIGINT,  lambda *_: (_on_exit(), exit(0)))
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_on_exit(), exit(0)))
    except Exception:
        pass  # Windows hat evtl. kein SIGTERM

    print(f"📂 captures: {CAP_DIR}")
    print(f"📝 aktive CSV: {os.path.basename(ACTIVE_CSV)}")
    print(f"📱 Handy:  http://<PC-IP>:{args.port}/  (z.B. http://192.168.1.13:{args.port}/)")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
