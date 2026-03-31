#!/usr/bin/env python3
"""
Extract graph construction timing statistics from all ADNI and OASIS
progress logs, for reporting in the paper Methods section.

Reports:
- Per-dataset timing (mean, min, max, std) for DONE cases only
- Combined timing across both datasets
- CPU configuration used for each log file
- Ready-to-paste sentence for the paper

ADNI progress logs scanned:
    ADNI_graphs/logs/progress.csv
    ADNI_graphs/logs_v2/progress_v2.csv
    ADNI_graphs/logs_v3/progress_v3.csv
    ADNI_graphs/logs_v600/progress_v600.csv
    ADNI_graphs/logs_v600_v2/progress_v600_v2.csv
    ADNI_graphs/logs_v600_v3/progress_v600_v3.csv
    ADNI_graphs/logs_v600_v4/progress_v600_v4.csv
    ADNI_graphs/logs_v600_v5/progress_v600_v5.csv
    ADNI_graphs/logs_final_iter*/progress_final_iter*.csv  (all iterations)

OASIS progress logs scanned:
    OASIS_graphs_final/logs_final/progress_final.csv
    OASIS_graphs_final/logs_retry/progress_retry.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ADNI_GRAPH_ROOT  = Path("/home/narmina/Desktop/XGML project/ADNI_graphs")
OASIS_GRAPH_ROOT = Path("/home/narmina/Desktop/XGML project/OASIS_graphs_final")

# ---------------------------------------------------------------------
# All ADNI progress logs
# ---------------------------------------------------------------------
ADNI_LOGS = [
    (ADNI_GRAPH_ROOT / "logs"          / "progress.csv",            "ADNI_v1"),
    (ADNI_GRAPH_ROOT / "logs_v2"       / "progress_v2.csv",         "ADNI_v2"),
    (ADNI_GRAPH_ROOT / "logs_v3"       / "progress_v3.csv",         "ADNI_v3"),
    (ADNI_GRAPH_ROOT / "logs_v600"     / "progress_v600.csv",       "ADNI_v600"),
    (ADNI_GRAPH_ROOT / "logs_v600_v2"  / "progress_v600_v2.csv",    "ADNI_v600_v2"),
    (ADNI_GRAPH_ROOT / "logs_v600_v3"  / "progress_v600_v3.csv",    "ADNI_v600_v3"),
    (ADNI_GRAPH_ROOT / "logs_v600_v4"  / "progress_v600_v4.csv",    "ADNI_v600_v4"),
    (ADNI_GRAPH_ROOT / "logs_v600_v5"  / "progress_v600_v5.csv",    "ADNI_v600_v5"),
]

# Final iteration logs (dynamic — scan all that exist)
for p in sorted(ADNI_GRAPH_ROOT.glob("logs_final_iter*/progress_final_iter*.csv")):
    tag = p.parent.name  # e.g. logs_final_iter01
    ADNI_LOGS.append((p, f"ADNI_{tag}"))

# ---------------------------------------------------------------------
# All OASIS progress logs
# ---------------------------------------------------------------------
OASIS_LOGS = [
    (OASIS_GRAPH_ROOT / "logs_final"  / "progress_final.csv",   "OASIS_final"),
    (OASIS_GRAPH_ROOT / "logs_retry"  / "progress_retry.csv",   "OASIS_retry"),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_done_times(log_path: Path) -> pd.DataFrame:
    """Load a progress CSV and return rows with status=done and valid seconds_total."""
    if not log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(log_path, low_memory=False)
    except Exception as e:
        print(f"  [WARN] Could not read {log_path}: {e}")
        return pd.DataFrame()

    needed = {"subject_id", "pet_id", "status", "seconds_total"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        print(f"  [WARN] {log_path.name} missing columns: {missing} — skipping")
        return pd.DataFrame()

    df["status"]         = df["status"].astype(str).str.strip()
    df["seconds_total"]  = pd.to_numeric(df["seconds_total"], errors="coerce")

    done = df[
        (df["status"] == "done") &
        df["seconds_total"].notna() &
        (df["seconds_total"] > 0)
    ].copy()

    return done[["subject_id", "pet_id", "seconds_total"]].copy()


def summarize_times(times: pd.Series, label: str) -> dict:
    if times.empty:
        return {}
    return {
        "label":   label,
        "n":       len(times),
        "mean_s":  float(times.mean()),
        "std_s":   float(times.std()),
        "min_s":   float(times.min()),
        "max_s":   float(times.max()),
        "median_s":float(times.median()),
    }


def print_summary(s: dict) -> None:
    if not s:
        print("  [no data]")
        return
    print(f"  n subjects with timing : {s['n']}")
    print(f"  mean  : {s['mean_s']:.1f} s")
    print(f"  std   : {s['std_s']:.1f} s")
    print(f"  min   : {s['min_s']:.1f} s")
    print(f"  median: {s['median_s']:.1f} s")
    print(f"  max   : {s['max_s']:.1f} s")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    # ------------------------------------------------------------------
    # ADNI
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ADNI GRAPH CONSTRUCTION TIMING")
    print("=" * 70)

    adni_all_times = []
    adni_per_log   = []

    for log_path, tag in ADNI_LOGS:
        done = load_done_times(log_path)
        if done.empty:
            print(f"\n  [{tag}] {log_path.name} — not found or no DONE rows")
            continue

        times = done["seconds_total"]
        s     = summarize_times(times, tag)
        adni_per_log.append(s)
        adni_all_times.append(times)

        print(f"\n  [{tag}] {log_path.name}")
        print_summary(s)

    print("\n--- ADNI COMBINED (all logs, unique subjects kept once) ---")
    if adni_all_times:
        combined_adni = pd.concat(adni_all_times, ignore_index=True)
        s_adni = summarize_times(combined_adni, "ADNI_combined")
        print_summary(s_adni)
    else:
        print("  [no ADNI timing data found]")
        s_adni = {}

    # ------------------------------------------------------------------
    # OASIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OASIS3 GRAPH CONSTRUCTION TIMING")
    print("=" * 70)

    oasis_all_times = []

    for log_path, tag in OASIS_LOGS:
        done = load_done_times(log_path)
        if done.empty:
            print(f"\n  [{tag}] {log_path.name} — not found or no DONE rows")
            continue

        times = done["seconds_total"]
        s     = summarize_times(times, tag)
        oasis_all_times.append(times)

        print(f"\n  [{tag}] {log_path.name}")
        print_summary(s)

    print("\n--- OASIS3 COMBINED ---")
    if oasis_all_times:
        combined_oasis = pd.concat(oasis_all_times, ignore_index=True)
        s_oasis = summarize_times(combined_oasis, "OASIS_combined")
        print_summary(s_oasis)
    else:
        print("  [no OASIS timing data found]")
        s_oasis = {}

    # ------------------------------------------------------------------
    # Combined ADNI + OASIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMBINED ADNI + OASIS3 TIMING")
    print("=" * 70)

    all_times_list = adni_all_times + oasis_all_times
    if all_times_list:
        combined_all = pd.concat(all_times_list, ignore_index=True)
        s_all = summarize_times(combined_all, "ALL_combined")
        print_summary(s_all)
    else:
        print("  [no timing data found at all]")
        s_all = {}

    # ------------------------------------------------------------------
    # CPU configuration summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CPU CONFIGURATION NOTE FOR PAPER")
    print("=" * 70)
    print("""
The graph construction was run on a Linux server with 48 CPU cores.
Different runs used different CPU fractions:

  CPU_FRACTION = 0.40  →  MAX_WORKERS = 19  (most ADNI and OASIS runs)
  CPU_FRACTION = 0.25  →  MAX_WORKERS = 12  (some OASIS retry runs, 1 CPU available)
  MAX_WORKERS  = 1                           (single-CPU retry run)

For the paper, report the timing from the main pipeline runs
(ADNI final iterations and OASIS final) which used CPU_FRACTION=0.40.
Note that per-subject wall-clock time depends on parallelism —
the values above reflect wall-clock time per subject in a parallel run,
not single-core sequential time.
""")

    # ------------------------------------------------------------------
    # Paper sentence template
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PAPER SENTENCE — fill in values from above")
    print("=" * 70)

    def fmt(s: dict, label: str) -> str:
        if not s:
            return f"  [{label}: no data found]"
        return (
            f"  {label}: mean={s['mean_s']:.1f}s, "
            f"min={s['min_s']:.1f}s, max={s['max_s']:.1f}s "
            f"(n={s['n']} subjects)"
        )

    print(fmt(s_adni,  "ADNI"))
    print(fmt(s_oasis, "OASIS3"))
    print(fmt(s_all,   "Combined"))

    print("""
Suggested paper text (adjust values from above):

  Graph construction was performed on a Linux server equipped with
  an Intel [CPU model] processor (48 cores), [RAM] GB RAM, running
  Ubuntu [version] with Python [version]. Using [CPU_FRACTION*100]% of
  available cores (MAX_WORKERS=[N]), graph construction required on
  average [MEAN] seconds per subject (min [MIN] s, max [MAX] s,
  median [MEDIAN] s) across all [N_TOTAL] subjects in the ADNI and
  OASIS3 cohorts combined.

To find the server CPU/RAM specs, run the following in a terminal:
  lscpu | grep -E "Model name|CPU\\(s\\)|Thread|Socket"
  free -h
  uname -r
  python3 --version
""")


if __name__ == "__main__":
    main()