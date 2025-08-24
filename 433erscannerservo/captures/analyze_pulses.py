#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
433 MHz Pulse Analyzer + Device Classifier
Eingang: Edges-CSV im Format (t_us, level, angle) – z. B. vom ESP32-Logger.

Features:
  • Pulsdauern (Δt) berechnen & plotten (gesamt / optional pro Winkel)
  • Pakete via Lücke (--gap in µs) segmentieren
  • „Signaturen“ je Paket (Hash, Länge, Avg/Min/Max, Short/Long-Ratio, Unit≈GCD)
  • Geräte-Heuristik: ordnet Signaturen zu (Remote/Funksteckdose, Wetter-/Haussensor, Türgong …)
  • Periodizität pro Signatur (Sekunden) + Wiederholungs-Bursts (Press-Repeats)
  • Optional: Ergebnis-CSV export

Python 3.7+  |  Benötigt: pandas, numpy, matplotlib
"""

import os, glob, argparse, hashlib, statistics, math
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------- Laden / CSV ----------------------------
def find_csv(path_arg: Optional[str]) -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.abspath(path_arg) if path_arg else base
    if os.path.isdir(p):
        cands = sorted(glob.glob(os.path.join(p, "rx_*.csv")), key=os.path.getmtime)
        if not cands:
            cands = sorted(glob.glob(os.path.join(p, "*.csv")), key=os.path.getmtime)
        return cands[-1] if cands else None
    return p if os.path.exists(p) else None

def sniff_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    return ";" if head.count(";") > head.count(",") else ","

def load_edges_csv(path: str) -> pd.DataFrame:
    sep = sniff_sep(path)
    df = pd.read_csv(path, sep=sep, header=None, names=["t_us","level","angle"], engine="python")
    df = df[pd.to_numeric(df["t_us"], errors="coerce").notna()]
    for c in ["t_us","level","angle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().astype({"t_us":"int64","level":"int64","angle":"int64"})
    df = df.sort_values("t_us").reset_index(drop=True)
    return df

# ---------------------------- Pulse & Pakete ----------------------------
def pulses_from_edges(df: pd.DataFrame, min_pulse_us: int = 50, max_pulse_us: int = 50000) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(columns=["pulse_us","angle","level"])
    dt = df["t_us"].diff().iloc[1:].astype(np.int64).values
    ang = df["angle"].iloc[1:].values
    lvl = df["level"].iloc[1:].values
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

# ---------------------------- Signaturen ----------------------------
def fingerprint_from_pulses(pulses_us: List[int]) -> Dict:
    if not pulses_us:
        return {}
    # quantisieren für Robustheit
    norm = [int(round(p/50.0)*50) for p in pulses_us]
    try:
        unit = int(np.gcd.reduce(np.array(norm, dtype=np.int64)))
    except Exception:
        unit = 0
    avg = statistics.mean(pulses_us)
    short = sum(1 for p in pulses_us if p < avg)
    long  = len(pulses_us) - short
    h = hashlib.sha1(str(norm).encode()).hexdigest()[:8]
    return {
        "hash": h,
        "len": len(pulses_us),
        "avg": round(avg, 1),
        "min": int(min(pulses_us)),
        "max": int(max(pulses_us)),
        "ratio": f"{short}/{long}",
        "unit": unit,
    }

def packet_meta(seg: pd.DataFrame) -> Tuple[int,int,int]:
    """liefert (t_start_us, angle_mode, edges_count) pro Paket."""
    t_start = int(seg["t_us"].iloc[0])
    # typischen Winkel (Modus) fürs Paket bestimmen
    ang = seg["angle"].mode()
    ang_mode = int(ang.iloc[0]) if not ang.empty else int(seg["angle"].iloc[0])
    return (t_start, ang_mode, len(seg))

# ---------------------------- Klassifikation ----------------------------
def classify_signature(unit_us: int, pkt_len: int, periods_s: List[float], burst_repeats: List[int]) -> Tuple[str, float, str]:
    """
    Heuristiken:
      • Fixed-code Remote / Funksteckdose (PT2262/EV1527-like):
          unit 250–500 µs, ratio ~1:3, Paketlänge ~40–120, Repeats je Press 3–10 in <0.2s,
          keine strikte Periodizität über viele Sekunden
      • Wetter-/Haussensor:
          unit 500–1200 µs, Paketlänge oft 100–400, periodisch (10–120s) mit kleiner Varianz,
          meist Wiederholungen 2–8 in ~100ms Abständen pro Sende-Ereignis
      • Doorbell/Car/Garage (rolling code):
          unit 200–600 µs, wenige Repeats (1–3), unregelmäßig, keine Periodizität
    """
    reason = []
    conf = 0.3
    label = "Unbekannt"

    # Periodizität evaluieren
    per = [p for p in periods_s if p > 1.0]  # ignoriere Subsekunden-Abstände (das sind Repeats)
    mean_per = np.mean(per) if per else None
    std_per  = np.std(per) if per else None
    is_periodic = bool(per) and mean_per >= 5 and mean_per <= 180 and (std_per / mean_per if mean_per else 1) < 0.25

    # Repeat-Verhalten
    if burst_repeats:
        mean_rep = np.mean(burst_repeats)
        max_rep  = np.max(burst_repeats)
    else:
        mean_rep = 1.0
        max_rep  = 1

    # Kandidat: Wetter-/Haussensor
    if (500 <= unit_us <= 1200) and (pkt_len >= 80) and is_periodic and (mean_rep >= 1.5):
        label = "Wetter-/Haussensor (periodisch)"
        conf = 0.85
        reason.append(f"unit≈{unit_us}µs (500–1200), len={pkt_len} (≥80), periodisch≈{mean_per:.1f}s (CV≈{(std_per/mean_per):.2f}), repeats≈{mean_rep:.1f}")
        return label, conf, "; ".join(reason)

    # Kandidat: Fixed-code Remote / Funksteckdose / Türgong
    if (250 <= unit_us <= 500) and (40 <= pkt_len <= 140) and (max_rep >= 3) and (not is_periodic):
        label = "Remote/Funksteckdose/Türgong (fixed-code)"
        conf = 0.75
        reason.append(f"unit≈{unit_us}µs (250–500), len={pkt_len} (40–140), repeats bis {int(max_rep)}, keine Periodizität")
        return label, conf, "; ".join(reason)

    # Kandidat: Auto/Garage (rolling-ish)
    if (200 <= unit_us <= 600) and (pkt_len <= 120) and (max_rep <= 3) and (not is_periodic):
        label = "Remote (mglw. rolling-code/Auto/Garage)"
        conf = 0.55
        reason.append(f"unit≈{unit_us}µs, len≤120, wenige Repeats ({int(max_rep)}), unregelmäßig")
        return label, conf, "; ".join(reason)

    # Unklar – Hinweise geben
    if is_periodic:
        reason.append(f"periodisch≈{mean_per:.1f}s (CV≈{(std_per/mean_per):.2f})")
        conf = 0.5
    reason.append(f"unit≈{unit_us}µs, len={pkt_len}, repeats≈{mean_rep:.1f}")
    return label, conf, "; ".join(reason)

def group_bursts(start_times_us: List[int], repeat_window_ms: int = 200) -> Tuple[List[List[int]], List[int], List[float]]:
    """
    Gruppiert Pakete zu „Bursts“ (z.B. eine Tastenbetätigung): Startzeiten, die näher als repeat_window_ms liegen.
    Liefert:
      • bursts: Liste von Gruppen (Startzeiten)
      • repeats_per_burst: Anzahl Pakete je Burst
      • periods_between_bursts_s: Sekundenabstände zwischen Burst-Starts
    """
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

# ---------------------------- Plots ----------------------------
def plot_hist(pulses: pd.DataFrame, by_angle: bool = False):
    plt.figure(figsize=(10,5))
    if by_angle:
        for ang, grp in pulses.groupby("angle"):
            if not grp.empty:
                plt.hist(grp["pulse_us"], bins=80, alpha=0.45, label=f"{ang}°")
        plt.legend(ncol=4, fontsize=8)
        plt.title("Pulsdauer-Histogramm (pro Winkel, überlagert)")
    else:
        plt.hist(pulses["pulse_us"], bins=100, alpha=0.85)
        plt.title("Pulsdauer-Histogramm (gesamt)")
    plt.xlabel("Pulsdauer [µs]")
    plt.ylabel("Häufigkeit")
    plt.grid(alpha=0.25)
    plt.tight_layout()

# ---------------------------- Hauptablauf ----------------------------
def main():
    ap = argparse.ArgumentParser(description="433 MHz Pulse Analyzer + Device Classifier")
    ap.add_argument("-p","--path", default=None, help="CSV-Datei oder Ordner (nimmt neueste rx_*.csv)")
    ap.add_argument("--gap", type=int, default=5000, help="Paketgrenze in µs (Default 5000)")
    ap.add_argument("--by-angle", action="store_true", help="Histogramm pro Winkel überlagert anzeigen")
    ap.add_argument("--angle", type=int, default=None, help="Nur Daten eines bestimmten Winkels auswerten")
    ap.add_argument("--min-pulse", type=int, default=50, help="untere Pulsdauer-Grenze in µs (Glitch-Filter)")
    ap.add_argument("--max-pulse", type=int, default=50000, help="obere Pulsdauer-Grenze in µs")
    ap.add_argument("--export", default=None, help="Ordner für Signatur-CSV-Exports (optional)")
    ap.add_argument("--summary-csv", default=None, help="Pfad für Geräte-Übersicht als CSV")
    ap.add_argument("--save", action="store_true", help="Plots zusätzlich als PNG speichern")
    args = ap.parse_args()

    csv_path = find_csv(args.path)
    if not csv_path:
        print("❌ Keine CSV gefunden.")
        return
    print("📄 CSV:", csv_path)

    df_e = load_edges_csv(csv_path)
    if args.angle is not None:
        df_e = df_e[df_e["angle"] == args.angle].reset_index(drop=True)
        print(f"↳ gefiltert auf Winkel {args.angle}°: {len(df_e)} Edges")
    if df_e.empty:
        print("❌ Keine Edges in Datei.")
        return

    pulses = pulses_from_edges(df_e, min_pulse_us=args.min_pulse, max_pulse_us=args.max_pulse)
    if pulses.empty:
        print("❌ Keine Pulse (nach Filter).")
        return

    # Plots
    plot_hist(pulses, by_angle=False)
    if args.by_angle:
        plot_hist(pulses, by_angle=True)

    # Pakete segmentieren
    packets = segment_packets(df_e, max_gap_us=args.gap)
    print(f"\n📦 Pakete erkannt: {len(packets)}  (gap={args.gap}µs)")

    # Signaturen sammeln + Zeitverhalten
    results = []  # für Summary
    sigs = {}
    for seg in packets:
        # pulses in diesem Segment
        p = seg["t_us"].diff().dropna().astype(np.int64)
        p = [x for x in p if x >= args.min_pulse and x <= args.max_pulse]
        if not p:
            continue
        fp = fingerprint_from_pulses(p)
        if not fp: 
            continue
        t_start, ang_mode, edges_cnt = packet_meta(seg)
        rec = sigs.setdefault(fp["hash"], {
            "meta": fp,
            "starts": [],
            "angles": [],
            "edges_cnt": [],
        })
        rec["starts"].append(t_start)
        rec["angles"].append(ang_mode)
        rec["edges_cnt"].append(edges_cnt)

        # optional export roher Pulslisten per Signatur
        if args.export:
            os.makedirs(args.export, exist_ok=True)
            out = os.path.join(args.export, f"signature_{fp['hash']}.csv")
            with open(out, "a", encoding="utf-8") as f:
                f.write(",".join(map(str, p)) + "\n")

    print("\n== Gerätesignaturen ==")
    for h, rec in sigs.items():
        m = rec["meta"]
        starts = rec["starts"]
        # Bursts & Perioden schätzen
        bursts, repeats_per_burst, periods_s = group_bursts(starts, repeat_window_ms=200)
        unit_us = int(m["unit"])
        pkt_len = int(m["len"])
        label, conf, why = classify_signature(unit_us, pkt_len, periods_s, repeats_per_burst)

        angles = rec["angles"]
        dom_angle = int(pd.Series(angles).mode().iloc[0]) if angles else None
        mean_period = float(np.mean([p for p in periods_s if p>1.0])) if periods_s else None

        print(f"🔑 {h} | Pakete={len(starts):3d} | Len={pkt_len:3d} | Unit≈{unit_us:4d}µs "
              f"| Angle≈{dom_angle if dom_angle is not None else '-':>3}° "
              f"| Period≈{f'{mean_period:.1f}s' if mean_period else '—'} "
              f"| Repeats/Burst≈{(np.mean(repeats_per_burst) if repeats_per_burst else 1):.1f}  -> {label}  (conf {conf:.2f})")
        print(f"    ↪ {why}")

        results.append({
            "hash": h,
            "packets": len(starts),
            "len": pkt_len,
            "unit_us": unit_us,
            "dominant_angle": dom_angle,
            "mean_period_s": round(mean_period,1) if mean_period else "",
            "mean_repeats": round(float(np.mean(repeats_per_burst)),1) if repeats_per_burst else 1.0,
            "label": label,
            "confidence": round(conf,2),
            "why": why
        })

    if args.summary_csv and results:
        outp = os.path.abspath(args.summary_csv)
        pd.DataFrame(results).to_csv(outp, index=False)
        print(f"\n💾 Zusammenfassung gespeichert: {outp}")

    if args.save:
        out_dir = os.path.dirname(csv_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, num in enumerate(plt.get_fignums(), 1):
            fig = plt.figure(num)
            path = os.path.join(out_dir, f"pulses_{ts}_{i:02d}.png")
            fig.savefig(path, dpi=150)
            print("💾 Plot gespeichert:", path)

    plt.show()

if __name__ == "__main__":
    main()
