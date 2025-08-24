#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
433-MHz Live Dashboard (NASA/Hacker Style) + Device Classifier (HUD)
Fix: HUD/Footer are created once and only updated (no overlapping/ghost texts).
"""

import os, glob, time, argparse
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

# ---------- Look & Feel ----------
plt.style.use("dark_background")
TITLE_FONT = dict(fontsize=14, weight="bold")
ACCENT  = "#35ffd2"
ACCENT2 = "#ff2e88"
ACCENT3 = "#18a0ff"
FIG_BG  = "#0b0f13"

# ---------- CSV discovery ----------
def find_latest_csv(path_arg: Optional[str]) -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.abspath(path_arg) if path_arg else base
    if os.path.isdir(p):
        pref = os.path.join(p, "signals.csv")
        if os.path.exists(pref):
            return pref
        cands = glob.glob(os.path.join(p, "*.csv"))
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)
    return p if os.path.exists(p) else None

def sniff_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    return ";" if head.count(";") > head.count(",") else ","

def load_csv_auto(path: str):
    sep = sniff_sep(path)
    # Events with header?
    try:
        df = pd.read_csv(path, sep=sep, engine="python")
        cols = [c.strip().lower() for c in df.columns]
        if set(["timestamp","angle","signal"]).issubset(cols):
            df = df.rename(columns={
                df.columns[cols.index("timestamp")]: "Timestamp",
                df.columns[cols.index("angle")]: "Angle",
                df.columns[cols.index("signal")]: "Signal"
            })[["Timestamp","Angle","Signal"]]
            for c in ["Timestamp","Angle","Signal"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().astype({"Timestamp":"int64","Angle":"int64","Signal":"int64"})
            return {"type":"events","df":df}
    except Exception:
        pass
    # Edges (3 columns, possibly no header)
    df2 = pd.read_csv(path, sep=sep, header=None, names=["t_us","level","angle"], engine="python")
    df2 = df2[pd.to_numeric(df2["t_us"], errors="coerce").notna()]
    if {"t_us","level","angle"}.issubset(df2.columns):
        df2["t_us"]  = pd.to_numeric(df2["t_us"],  errors="coerce")
        df2["level"] = pd.to_numeric(df2["level"], errors="coerce")
        df2["angle"] = pd.to_numeric(df2["angle"], errors="coerce")
        df2 = df2.dropna().astype({"t_us":"int64","level":"int64","angle":"int64"})
        df2 = df2.sort_values("t_us").reset_index(drop=True)
        return {"type":"edges","df":df2}
    raise ValueError("CSV format not recognized (events or edges).")

# ---------- Edge→Pulse & packets ----------
def pulses_from_edges(df_edges: pd.DataFrame, min_pulse_us: int = 50, max_pulse_us: int = 50000) -> pd.DataFrame:
    if len(df_edges) < 2:
        return pd.DataFrame(columns=["pulse_us","angle","level"])
    dt = df_edges["t_us"].diff().iloc[1:].astype(np.int64).values
    ang = df_edges["angle"].iloc[1:].values
    lvl = df_edges["level"].iloc[1:].values
    out = pd.DataFrame({"pulse_us": dt, "angle": ang, "level": lvl})
    return out[(out["pulse_us"] >= min_pulse_us) & (out["pulse_us"] <= max_pulse_us)].reset_index(drop=True)

def segment_packets(df_edges: pd.DataFrame, max_gap_us: int = 5000, min_edges: int = 6) -> List[pd.DataFrame]:
    segs = []
    if df_edges.empty:
        return segs
    t = df_edges["t_us"].values
    start = 0
    for i in range(1, len(df_edges)):
        if (t[i] - t[i-1]) > max_gap_us:
            if i - start >= min_edges:
                segs.append(df_edges.iloc[start:i].copy())
            start = i
    if len(df_edges) - start >= min_edges:
        segs.append(df_edges.iloc[start:].copy())
    return segs

def fingerprint_from_pulses(pulses_us: List[int]) -> Dict:
    if not pulses_us:
        return {}
    norm = [int(round(p/50.0)*50) for p in pulses_us]
    try:
        unit = int(np.gcd.reduce(np.array(norm, dtype=np.int64)))
    except Exception:
        unit = 0
    avg = float(np.mean(pulses_us))
    short = sum(1 for p in pulses_us if p < avg)
    long  = len(pulses_us) - short
    return {
        "len": len(pulses_us),
        "avg": round(avg, 1),
        "min": int(min(pulses_us)),
        "max": int(max(pulses_us)),
        "ratio": "%d/%d" % (short, long),
        "unit": int(unit),
    }

def group_bursts(start_times_us: List[int], repeat_window_ms: int = 200):
    if not start_times_us:
        return [], [], []
    start_times_us = sorted(start_times_us)
    bursts = []
    current = [start_times_us[0]]
    for t in start_times_us[1:]:
        if (t - current[-1]) <= repeat_window_ms * 1000:
            current.append(t)
        else:
            bursts.append(current)
            current = [t]
    bursts.append(current)
    repeats_per_burst = [len(b) for b in bursts]
    periods_s = []
    for i in range(1, len(bursts)):
        periods_s.append((bursts[i][0] - bursts[i-1][0]) / 1_000_000.0)
    return bursts, repeats_per_burst, periods_s

def classify_signature(unit_us: int, pkt_len: int, periods_s: List[float], burst_repeats: List[int]) -> Tuple[str, float, str]:
    reason = []
    conf = 0.3
    label = "Unknown"

    per = [p for p in periods_s if p > 1.0]
    mean_per = np.mean(per) if per else None
    std_per  = np.std(per) if per else None
    cv = (std_per/mean_per) if mean_per else 1.0
    is_periodic = bool(per) and (5 <= mean_per <= 180) and (cv < 0.25)

    mean_rep = float(np.mean(burst_repeats)) if burst_repeats else 1.0
    max_rep  = int(np.max(burst_repeats)) if burst_repeats else 1

    if (500 <= unit_us <= 1200) and (pkt_len >= 80) and is_periodic and (mean_rep >= 1.5):
        label = "Weather/Home sensor (periodic)"
        conf = 0.85
        reason.append(f"unit≈{unit_us}µs (500–1200), len={pkt_len}≥80, period≈{mean_per:.1f}s (CV≈{cv:.2f}), repeats≈{mean_rep:.1f}")
        return label, conf, "; ".join(reason)

    if (250 <= unit_us <= 500) and (40 <= pkt_len <= 140) and (max_rep >= 3) and (not is_periodic):
        label = "Remote/Smart plug/Doorbell (fixed-code)"
        conf = 0.75
        reason.append(f"unit≈{unit_us}µs (250–500), len={pkt_len} (40–140), repeats≳{max_rep}, non-periodic")
        return label, conf, "; ".join(reason)

    if (200 <= unit_us <= 600) and (pkt_len <= 120) and (max_rep <= 3) and (not is_periodic):
        label = "Remote (possibly rolling code/Car/Garage)"
        conf = 0.55
        reason.append(f"unit≈{unit_us}µs, len≤120, repeats≤{max_rep}, irregular")
        return label, conf, "; ".join(reason)

    if is_periodic:
        reason.append(f"period≈{mean_per:.1f}s (CV≈{cv:.2f})")
        conf = 0.5
    reason.append(f"unit≈{unit_us}µs, len={pkt_len}, repeats≈{mean_rep:.1f}")
    return label, conf, "; ".join(reason)

# ---------- Heatmaps + colorbar handling ----------
_CBAR_REGISTRY: Dict[int, Dict[int, object]] = {}
def _remove_cbar(fig, ax):
    fid = id(fig); aid = id(ax)
    if fid in _CBAR_REGISTRY and aid in _CBAR_REGISTRY[fid]:
        try:
            _CBAR_REGISTRY[fid][aid].remove()
        except Exception:
            pass
        del _CBAR_REGISTRY[fid][aid]

def _replace_cbar(fig, ax, im, label=""):
    _remove_cbar(fig, ax)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    if label: cb.set_label(label)
    _CBAR_REGISTRY.setdefault(id(fig), {})[id(ax)] = cb

def build_heatmap_events(ax, df, bin_seconds=30, fig=None):
    hits = df[df["Signal"]==1].copy()
    ax.clear()
    if hits.empty:
        _remove_cbar(fig, ax)
        ax.text(0.5,0.5,"No hits for heatmap", ha="center", va="center", color="#bbb", transform=ax.transAxes)
        ax.set_title(f"Heatmap (events) – bin={bin_seconds}s", color=ACCENT); return
    t0 = hits["Timestamp"].min()
    hits["t_sec"] = (hits["Timestamp"] - t0)/1000.0
    max_t = hits["t_sec"].max()
    nb = max(1, int(np.ceil(max_t/bin_seconds)))
    edges = np.linspace(0, nb*bin_seconds, nb+1)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    hits["bin"] = pd.cut(hits["t_sec"], bins=edges, labels=labels, include_lowest=True, right=False)
    pivot = hits.pivot_table(index="Angle", columns="bin", values="Signal", aggfunc="count", fill_value=0).sort_index()
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   extent=[0, pivot.shape[1], pivot.index.min(), pivot.index.max()],
                   cmap="viridis")
    _replace_cbar(fig, ax, im, label="Hit count")
    ax.set_title(f"Heatmap (events) – bin={bin_seconds}s", color=ACCENT)
    ax.set_xlabel("Time window (s)"); ax.set_ylabel("Angle (°)")
    ax.set_xticks(np.arange(pivot.shape[1]) + 0.5)
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)

def build_heatmap_edges(ax, df_edges, bin_seconds=30, fig=None):
    ax.clear()
    if df_edges.empty:
        _remove_cbar(fig, ax)
        ax.text(0.5,0.5,"No edges for heatmap", ha="center", va="center", color="#bbb", transform=ax.transAxes)
        ax.set_title(f"Heatmap (edges) – bin={bin_seconds}s", color=ACCENT); return
    t0 = int(df_edges["t_us"].min())
    df = df_edges.copy()
    df["t_sec"] = (df["t_us"] - t0)/1_000_000.0
    max_t = df["t_sec"].max()
    nb = max(1, int(np.ceil(max_t/bin_seconds)))
    edges = np.linspace(0, nb*bin_seconds, nb+1)
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    df["bin"] = pd.cut(df["t_sec"], bins=edges, labels=labels, include_lowest=True, right=False)
    pivot = df.pivot_table(index="angle", columns="bin", values="level", aggfunc="count", fill_value=0).sort_index()
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   extent=[0, pivot.shape[1], pivot.index.min(), pivot.index.max()],
                   cmap="magma")
    _replace_cbar(fig, ax, im, label="Edges")
    ax.set_title(f"Heatmap (edges) – bin={bin_seconds}s", color=ACCENT)
    ax.set_xlabel("Time window (s)"); ax.set_ylabel("Angle (°)")
    ax.set_xticks(np.arange(pivot.shape[1]) + 0.5)
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)

# ---------- Live dashboard ----------
class LiveDashboard:
    def __init__(self, path: Optional[str], interval: float = 1.0, bins: int = 30,
                 save_every: int = 0, bg_path: Optional[str] = None, bg_alpha: float = 0.18,
                 classify: bool = False, gap_us: int = 8000, classify_window_s: int = 45):
        self.interval = max(0.5, float(interval))
        self.bins = int(bins)
        self.save_every = int(save_every)
        self.update_counter = 0

        self.classify = bool(classify)
        self.gap_us = int(gap_us)
        self.classify_window_s = int(classify_window_s)

        self.dir_default = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = find_latest_csv(path or self.dir_default)
        if not self.csv_path:
            raise FileNotFoundError("No CSV found.")
        self.last_sig = (0, 0)  # (mtime, size)

        # Figure & background
        self.fig = plt.figure(figsize=(16,9), facecolor=FIG_BG)
        self.bg_ax = self.fig.add_axes([0,0,1,1], label="bg", zorder=0)
        self.bg_ax.set_axis_off()
        self._set_background_image(bg_path, bg_alpha)

        # Content
        self.gs  = GridSpec(2, 2, figure=self.fig, height_ratios=[1,1.05],
                            width_ratios=[1,1], hspace=0.25, wspace=0.18)
        self.axA = self.fig.add_subplot(self.gs[0,0], label="A", zorder=2)  # line
        self.axB = self.fig.add_subplot(self.gs[0,1], projection="polar", label="B", zorder=2)
        self.axC = self.fig.add_subplot(self.gs[1,0], label="C", zorder=2)  # heatmap
        self.axD = self.fig.add_subplot(self.gs[1,1], label="D", zorder=2)  # pulse spectrum
        self.fig.suptitle("433 MHz – Direction & Signal Dashboard (LIVE)", color=ACCENT, **TITLE_FONT)

        # HUD + footer (created once)
        self.hud = self.fig.text(
            0.5, 0.945, "", ha="center", va="center",
            color=ACCENT, fontsize=12, weight="bold",
            bbox=dict(facecolor=(0,0,0,0.35), edgecolor=ACCENT, boxstyle="round,pad=0.35")
        )
        self.footer_left  = self.fig.text(0.01, 0.01, "", color="#9fb0bf", fontsize=9, ha="left")
        self.footer_right = self.fig.text(0.99, 0.01, "", color="#9fb0bf", fontsize=9, ha="right")

    # Background
    def _set_background_image(self, path: Optional[str], alpha: float):
        self.bg_ax.clear()
        if path and os.path.exists(path):
            try:
                img = mpimg.imread(path)
                self.bg_ax.imshow(img, aspect="auto", extent=[0,1,0,1], alpha=float(alpha), zorder=0)
            except Exception as e:
                print("⚠️ Failed to load background image:", e)
        self.bg_ax.set_axis_off()

    # HUD / footer update
    def _update_hud(self, text: str):
        self.hud.set_text(text or "")

    def _update_footer(self, left: str, right: str):
        self.footer_left.set_text(left or "")
        self.footer_right.set_text(right or "")

    def _classify_edges_window(self, df_edges: pd.DataFrame) -> str:
        if df_edges.empty:
            return "— no edges —"
        # last N seconds
        t_last = int(df_edges["t_us"].max())
        if self.classify_window_s > 0:
            t_min = t_last - self.classify_window_s * 1_000_000
            df_edges = df_edges[df_edges["t_us"] >= t_min].reset_index(drop=True)
        if df_edges.empty:
            return f"— no signal in the last {self.classify_window_s}s —"

        # packets
        packets = segment_packets(df_edges, max_gap_us=self.gap_us, min_edges=6)
        if not packets:
            return "— no packets in window —"

        # per packet fingerprint & start times
        by_sig: Dict[str, Dict] = {}
        for seg in packets:
            pulses = seg["t_us"].diff().dropna().astype(np.int64).tolist()
            pulses = [p for p in pulses if p >= 50 and p <= 50000]
            if not pulses:
                continue
            fp = fingerprint_from_pulses(pulses)
            key = f"len{fp['len']}_u{fp['unit']}"
            rec = by_sig.setdefault(key, {"meta": fp, "starts": [], "angles": []})
            rec["starts"].append(int(seg["t_us"].iloc[0]))
            rec["angles"].append(int(seg["angle"].mode().iloc[0]))

        if not by_sig:
            return "— no usable packets —"

        # best signature = most starts
        best_key = max(by_sig.keys(), key=lambda k: len(by_sig[k]["starts"]))
        rec = by_sig[best_key]
        meta = rec["meta"]
        starts = rec["starts"]
        angles = rec["angles"]

        bursts, repeats_per_burst, periods_s = group_bursts(starts, repeat_window_ms=200)
        unit_us = int(meta["unit"]); pkt_len = int(meta["len"])
        label, conf, why = classify_signature(unit_us, pkt_len, periods_s, repeats_per_burst)
        dom_angle = int(pd.Series(angles).mode().iloc[0]) if angles else None
        mean_period = float(np.mean([p for p in periods_s if p>1.0])) if periods_s else None
        rep_mean = float(np.mean(repeats_per_burst)) if repeats_per_burst else 1.0

        hud = f"{label}  (conf {conf:.2f}) | Unit≈{unit_us}µs  Len={pkt_len}  " \
              f"Period≈{(f'{mean_period:.1f}s' if mean_period else '—')}  " \
              f"Repeats≈{rep_mean:.1f}×  @Angle≈{dom_angle if dom_angle is not None else '—'}°"
        return hud

    # Draw one frame
    def _draw(self, meta):
        kind, df = meta["type"], meta["df"]

        # A) line
        self.axA.clear()
        if kind == "events":
            hits = df[df["Signal"]==1]
            ac = hits.groupby("Angle")["Signal"].count().sort_index()
            if len(ac):
                best = int(ac.idxmax())
                self.axA.plot(ac.index, ac.values, marker="o", lw=1.5, color=ACCENT3)
                self.axA.axvline(best, ls="--", color=ACCENT2, lw=1.2, label=f"Best {best}°")
                self.axA.legend(loc="upper right")
                self.axA.set_ylabel("Hits (Signal==1)")
            else:
                self.axA.text(0.5,0.5,"No hits", ha="center", va="center", color="#bbb", transform=self.axA.transAxes)
            self.axA.set_xlabel("Angle (°)"); self.axA.set_title("Hits per angle", color=ACCENT); self.axA.grid(alpha=0.25)
        else:
            counts = df.groupby("angle")["level"].count().sort_index()
            if len(counts):
                best = int(counts.idxmax())
                self.axA.plot(counts.index, counts.values, marker="o", lw=1.5, color=ACCENT3)
                self.axA.axvline(best, ls="--", color=ACCENT2, lw=1.2, label=f"Max {best}°")
                self.axA.legend(loc="upper right")
            else:
                self.axA.text(0.5,0.5,"No edges", ha="center", va="center", color="#bbb", transform=self.axA.transAxes)
            self.axA.set_ylabel("Edges"); self.axA.set_xlabel("Angle (°)")
            self.axA.set_title("Edges per angle", color=ACCENT); self.axA.grid(alpha=0.25)

        # B) polar
        self.axB.clear()
        if kind == "events":
            ac = df[df["Signal"]==1].groupby("Angle")["Signal"].count().sort_index()
            theta = np.deg2rad(ac.index.values.astype(float)) if len(ac) else np.array([0.0])
            r = ac.values.astype(float) if len(ac) else np.array([0.0])
        else:
            c = df.groupby("angle")["level"].count().sort_index()
            theta = np.deg2rad(c.index.values.astype(float)) if len(c) else np.array([0.0])
            r = c.values.astype(float) if len(c) else np.array([0.0])
        self.axB.plot(theta, r, marker="o", lw=1.2, color=ACCENT3)
        self.axB.fill(theta, r, alpha=0.18, color=ACCENT3)
        self.axB.set_theta_zero_location("N"); self.axB.set_theta_direction(-1)
        self.axB.set_thetagrids(range(0,360,30))
        self.axB.set_title("Direction radar", color=ACCENT)

        # C) heatmap
        if kind == "events":
            build_heatmap_events(self.axC, df, bin_seconds=self.bins, fig=self.fig)
        else:
            build_heatmap_edges(self.axC, df, bin_seconds=self.bins, fig=self.fig)

        # D) pulse spectrum & HUD (edges only)
        self.axD.clear()
        hud_text = ""
        if kind == "edges":
            pulses = pulses_from_edges(df)
            if not pulses.empty:
                self.axD.hist(pulses["pulse_us"], bins=120, color=ACCENT2, alpha=0.85)
                self.axD.set_xlabel("Pulse width [µs]"); self.axD.set_ylabel("Count")
                self.axD.set_title("Pulse spectrum (histogram)", color=ACCENT); self.axD.grid(alpha=0.2)
                med = int(np.median(pulses["pulse_us"]))
                self.axD.axvline(med, ls="--", lw=1.2, color=ACCENT, label=f"Median ≈ {med}µs")
                self.axD.legend(loc="upper right")
            else:
                self.axD.text(0.5,0.5,"No pulses detected", ha="center", va="center", color="#bbb", transform=self.axD.transAxes)

            if self.classify:
                try:
                    hud_text = self._classify_edges_window(df)
                except Exception as e:
                    hud_text = f"Classifier error: {e}"
        else:
            self.axD.text(0.5,0.5,"Pulse spectrum only with Edges CSV", ha="center", va="center", color="#bbb", transform=self.axD.transAxes)

        # Footer & HUD
        csv_name = os.path.basename(self.csv_path)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._update_footer(left=f"Source: {csv_name}", right=ts)
        self._update_hud(hud_text)

        plt.pause(0.001)

        # optional screenshot + "latest"
        self.update_counter += 1
        if self.save_every and (self.update_counter % self.save_every == 0):
            out_dir = os.path.dirname(self.csv_path)
            out = os.path.join(out_dir, f"dashboard_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            self.fig.savefig(out, dpi=160)
            latest = os.path.join(out_dir, "dashboard_latest.png")
            self.fig.savefig(latest, dpi=140)  # fixed file for phone viewer

    def _file_signature(self, path):
        try:
            st = os.stat(path)
            return (int(st.st_mtime), int(st.st_size))
        except FileNotFoundError:
            return (0, 0)

    def run(self):
        print("📡 Live dashboard running. Press CTRL+C to exit.")
        while True:
            latest = find_latest_csv(self.dir_default)
            if latest:
                self.csv_path = latest
            sig = self._file_signature(self.csv_path)
            if sig != self.last_sig:
                self.last_sig = sig
                try:
                    meta = load_csv_auto(self.csv_path)
                    print(f"↻ Update {datetime.now().strftime('%H:%M:%S')}  "
                          f"({os.path.basename(self.csv_path)}, {sig[1]} bytes, {meta['type']})")
                    self._draw(meta)
                except Exception as e:
                    print("⚠️ Problem while loading/rendering:", e)
            time.sleep(self.interval)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="433-MHz Live Dashboard (CSV polling) with background & device classifier")
    ap.add_argument("-p","--path", default=None,
                    help="File or folder. Default: this script's folder (uses signals.csv or newest .csv).")
    ap.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default 1.0)")
    ap.add_argument("--bins", type=int, default=30, help="Heatmap bin size in seconds")
    ap.add_argument("--save-every", type=int, default=0, help="Save a screenshot every N updates (0=off)")
    ap.add_argument("--bg", type=str, default=None, help="Path to background image (PNG/JPG)")
    ap.add_argument("--bg-alpha", type=float, default=0.18, help="Background image opacity (0..1)")
    # Classification
    ap.add_argument("--classify", action="store_true", help="Enable device classifier (edges CSV only)")
    ap.add_argument("--gap", type=int, default=8000, help="Packet gap for classifier in µs (default 8000)")
    ap.add_argument("--classify-window", type=int, default=45, help="Window in seconds for classifier (default 45)")
    args = ap.parse_args()

    try:
        live = LiveDashboard(args.path, interval=args.interval, bins=args.bins,
                             save_every=args.save_every, bg_path=args.bg, bg_alpha=args.bg_alpha,
                             classify=args.classify, gap_us=args.gap, classify_window_s=args.classify_window)
        live.run()
    except FileNotFoundError as e:
        print("❌", e)

if __name__ == "__main__":
    main()
