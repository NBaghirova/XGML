#!/usr/bin/env python3
"""
ADNI Metabolic Brain Graph Construction
========================================
Constructs subject-level metabolic brain graphs from FDG-PET scans using:
  - Kernel Density Estimation (KDE) per ROI via FFTKDE with ISJ bandwidth
  - Dynamic Time Warping (DTW) pairwise distances between ROI KDE curves
  - Schaefer 2018 200-parcel atlas (100 LH + 100 RH parcels)

The output per subject is:
  - A 200x200 symmetric distance matrix (.npy)
  - The upper-triangle feature vector of length 19,900 (.npy)
  - A CSV listing all pairwise distances (.csv)

Input CSV format
----------------
The cohort CSV must contain at least these columns:
    subject_id   : unique subject identifier
    pet_id       : unique PET scan identifier
    pet_file     : absolute path to the preprocessed FDG-PET NIfTI file
                   (expects SUVR-normalised .nii or .nii.gz)
    lh_atlas     : absolute path to the left-hemisphere Schaefer200 atlas
                   (NIfTI, labels 1..100)
    rh_atlas     : absolute path to the right-hemisphere Schaefer200 atlas
                   (NIfTI, labels 1..100)

Usage
-----
1. Fill in the CONFIGURATION section below with your paths.
2. Run:  python adni_graph_construction.py

Resume behaviour
----------------
If all three output files for a subject already exist in OUTPUT_DIR, that
subject is skipped and logged as "skipped_exists". This allows interrupted
runs to be resumed without recomputing completed subjects. To force a full
recomputation, delete the relevant subject folders inside OUTPUT_DIR.

Requirements
------------
See requirements.txt

Reference
---------
If you use this code, please cite:
    Explainable Graph-theoretical Machine Learning: with Application to Alzheimer's Disease Prediction
	Authors: Narmina Baghirova, Duy-Thanh Vũ, Duy-Cat Can, Christelle Schneuwly Diaz, Julien Bodlet, Guillaume Blanc, Georgi Hrusanov, Bernard Ries, Oliver Y. Chén

ADNI data access
----------------
Data used in preparation of this script were obtained from the
Alzheimer's Disease Neuroimaging Initiative (ADNI) database
(https://adni.loni.usc.edu). ADNI data are not distributed with
this code. Researchers must apply for access at adni.loni.usc.edu.
"""

from __future__ import annotations

import os
import csv
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# CONFIGURATION — fill in your paths here
# =============================================================================

# Path to your cohort CSV (see Input CSV format above)
COHORT_CSV = Path("*** INSERT PATH TO YOUR COHORT CSV ***")
# Example: COHORT_CSV = Path("/data/adni/cohort_600.csv")

# Directory where per-subject graph files will be written
OUTPUT_DIR = Path("*** INSERT PATH TO YOUR OUTPUT DIRECTORY ***")
# Example: OUTPUT_DIR = Path("/data/adni/graphs")

# Fraction of available CPU cores to use (0.0–1.0)
# E.g. 0.40 uses 40% of cores. Adjust to suit your system load.
CPU_FRACTION = 0.40

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# CPU / thread control — set before importing numpy
# -----------------------------------------------------------------------------
TOTAL_CPUS  = os.cpu_count() or 1
MAX_WORKERS = max(1, int(TOTAL_CPUS * CPU_FRACTION))

os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import nibabel as nib
from KDEpy import FFTKDE
from dtw import dtw

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KDE_GRID_N         = 2 ** 10   # KDE evaluation grid size (FFT-friendly)
KDE_MARGIN         = 0.3       # margin added to each side of the KDE range
PREFER_OVERLAP     = "lh"      # which hemisphere wins in the rare overlap voxels
EXPECTED_ROIS      = 200       # total parcels (100 LH + 100 RH)
EXPECTED_LH_LABELS = 100
EXPECTED_RH_LABELS = 100
MAX_OVERLAP_FRAC   = 0.01      # max tolerated LH/RH voxel overlap fraction

REQUIRED_COLS = ["subject_id", "pet_id", "pet_file", "lh_atlas", "rh_atlas"]


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def init_progress_log(progress_csv: Path) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    if not progress_csv.exists():
        with open(progress_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "subject_id", "pet_id", "status",
                "seconds_total", "seconds_load", "seconds_atlasmerge",
                "seconds_kde", "seconds_dtw",
                "out_dir", "dist_csv", "dist_npy", "feat_npy",
                "n_rois", "overlap_vox", "note",
            ])


def append_log(progress_csv: Path, row: list) -> None:
    with open(progress_csv, "a", newline="") as f:
        csv.writer(f).writerow(row)


# -----------------------------------------------------------------------------
# NIfTI / atlas helpers
# -----------------------------------------------------------------------------
def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img  = nib.load(str(path))
    data = img.get_fdata()
    return data, img


def unique_nonzero_int_labels(arr: np.ndarray) -> np.ndarray:
    labs = np.unique(arr.astype(np.int32, copy=False))
    labs = labs[labs != 0]
    return np.array(sorted(labs.tolist()), dtype=np.int32)


def combine_lh_rh_atlases_safe(
    lh_arr: np.ndarray,
    rh_arr: np.ndarray,
    prefer: str = "lh",
) -> tuple[np.ndarray, dict]:
    """
    Merge left- and right-hemisphere atlas volumes into a single 200-label atlas.

    LH labels stay as-is (1..100).
    RH labels are offset by the LH maximum so they become 101..200.
    Voxels that appear in both hemispheres (boundary artifacts) are resolved
    in favour of the hemisphere specified by `prefer`.
    """
    if lh_arr.shape != rh_arr.shape:
        raise ValueError(
            f"Atlas shape mismatch: lh={lh_arr.shape}, rh={rh_arr.shape}"
        )

    lh = lh_arr.astype(np.int32, copy=True)
    rh = rh_arr.astype(np.int32, copy=True)

    lh_labels = unique_nonzero_int_labels(lh)
    rh_labels = unique_nonzero_int_labels(rh)

    if (len(lh_labels) != EXPECTED_LH_LABELS
            or lh_labels.min() != 1
            or lh_labels.max() != EXPECTED_LH_LABELS):
        raise RuntimeError(
            f"Unexpected LH labels: n={len(lh_labels)}, "
            f"min={lh_labels.min() if len(lh_labels) else 'NA'}, "
            f"max={lh_labels.max() if len(lh_labels) else 'NA'} "
            f"(expected 1..{EXPECTED_LH_LABELS})"
        )

    if (len(rh_labels) != EXPECTED_RH_LABELS
            or rh_labels.min() != 1
            or rh_labels.max() != EXPECTED_RH_LABELS):
        raise RuntimeError(
            f"Unexpected RH labels: n={len(rh_labels)}, "
            f"min={rh_labels.min() if len(rh_labels) else 'NA'}, "
            f"max={rh_labels.max() if len(rh_labels) else 'NA'} "
            f"(expected 1..{EXPECTED_RH_LABELS})"
        )

    lh_mask = lh > 0
    rh_mask = rh > 0
    ov      = lh_mask & rh_mask

    n_lh       = int(lh_mask.sum())
    n_rh       = int(rh_mask.sum())
    n_ov       = int(ov.sum())
    ov_frac_lh = n_ov / max(1, n_lh)
    ov_frac_rh = n_ov / max(1, n_rh)

    if ov_frac_lh > MAX_OVERLAP_FRAC or ov_frac_rh > MAX_OVERLAP_FRAC:
        raise RuntimeError(
            f"Voxel overlap too large: overlap_vox={n_ov}, "
            f"frac_lh={ov_frac_lh:.4f}, frac_rh={ov_frac_rh:.4f}. "
            "Atlas generation likely incorrect."
        )

    if n_ov > 0:
        if prefer == "lh":
            rh[ov] = 0
        else:
            lh[ov] = 0

    lh_max = int(lh.max())
    if lh_max <= 0:
        raise RuntimeError("LH atlas has no positive labels after overlap resolution.")

    rh_offset            = rh.copy()
    rh_offset[rh_offset > 0] += lh_max

    combined             = lh.copy()
    combined[rh_offset > 0] = rh_offset[rh_offset > 0]

    comb_labels = unique_nonzero_int_labels(combined)
    if (len(comb_labels) != EXPECTED_ROIS
            or comb_labels.min() != 1
            or comb_labels.max() != EXPECTED_ROIS):
        raise RuntimeError(
            f"Combined atlas labels unexpected: n={len(comb_labels)}, "
            f"min={comb_labels.min() if len(comb_labels) else 'NA'}, "
            f"max={comb_labels.max() if len(comb_labels) else 'NA'} "
            f"(expected 1..{EXPECTED_ROIS})"
        )

    return combined, {
        "lh_vox": n_lh, "rh_vox": n_rh, "overlap_vox": n_ov,
        "overlap_frac_lh": ov_frac_lh, "overlap_frac_rh": ov_frac_rh,
    }


def extract_region_values(
    image_data: np.ndarray,
    atlas_labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    """Return per-ROI finite PET voxel values and voxel counts."""
    if image_data.shape != atlas_labels.shape:
        raise ValueError(
            f"Shape mismatch: image={image_data.shape}, atlas={atlas_labels.shape}"
        )

    labels = unique_nonzero_int_labels(atlas_labels)
    if len(labels) != EXPECTED_ROIS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROIS} ROI labels in atlas, got {len(labels)}"
        )

    values_by_label: dict[int, np.ndarray] = {}
    voxel_counts:    dict[int, int]        = {}

    for lab in labels:
        mask = (atlas_labels == lab)
        cnt  = int(mask.sum())
        voxel_counts[int(lab)] = cnt
        if cnt == 0:
            raise RuntimeError(f"ROI {int(lab)} has 0 voxels in atlas.")

        vals = image_data[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise RuntimeError(
                f"ROI {int(lab)} has no finite PET voxels (all NaN/Inf)."
            )
        values_by_label[int(lab)] = vals.astype(np.float64, copy=False)

    return labels, values_by_label, voxel_counts


# -----------------------------------------------------------------------------
# KDE
# -----------------------------------------------------------------------------
def kde_density(values: np.ndarray, roi_label: int | None = None) -> np.ndarray:
    """
    Estimate the probability density of PET values within one ROI using
    FFT-based KDE with Improved Sheather-Jones (ISJ) bandwidth selection.

    Parameters
    ----------
    values    : 1-D array of finite PET voxel intensities for this ROI
    roi_label : ROI index (used only for error messages)

    Returns
    -------
    density   : 1-D array of length KDE_GRID_N
    """
    if values.size == 0:
        raise RuntimeError(f"KDE failure for ROI {roi_label}: empty values.")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"KDE failure for ROI {roi_label}: non-finite values.")

    vmin0, vmax0 = float(values.min()), float(values.max())

    if vmin0 == vmax0:
        raise RuntimeError(
            f"KDE failure for ROI {roi_label}: constant values "
            f"(min=max={vmin0:.6g}). KDE requires variance."
        )

    vmin = vmin0 - KDE_MARGIN
    vmax = vmax0 + KDE_MARGIN

    x = np.linspace(vmin, vmax, KDE_GRID_N).reshape(-1, 1)
    v = values.reshape(-1, 1)

    try:
        dens = FFTKDE(kernel="gaussian", bw="ISJ").fit(v).evaluate(x)
    except Exception as e:
        raise RuntimeError(
            f"KDE failure for ROI {roi_label}: {type(e).__name__}: {e}"
        ) from e

    if dens is None or len(dens) != KDE_GRID_N:
        raise RuntimeError(f"KDE failure for ROI {roi_label}: unexpected output length.")
    if not np.all(np.isfinite(dens)):
        raise RuntimeError(f"KDE failure for ROI {roi_label}: NaN/Inf in density.")

    return dens.astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# DTW
# -----------------------------------------------------------------------------
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two KDE curves using Euclidean local cost."""
    alignment = dtw(a, b, dist_method="euclidean", keep_internals=False)
    d = float(alignment.distance)
    if not np.isfinite(d):
        raise RuntimeError("DTW produced non-finite distance.")
    return d


def upper_triangle_vector(mat: np.ndarray) -> np.ndarray:
    """Return the upper-triangle (k=1) of a square matrix as a 1-D vector."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu].astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# Per-subject graph construction (runs in a worker process)
# -----------------------------------------------------------------------------
def process_one_graph(row_dict: dict) -> tuple[list, dict | None]:
    """
    Build the metabolic distance graph for one subject.

    Returns
    -------
    log_row     : list of fields to append to the progress CSV
    success_row : dict with output file paths (None if construction failed)
    """
    sid      = str(row_dict["subject_id"]).strip()
    pid      = str(row_dict["pet_id"]).strip()
    pet_file = str(row_dict["pet_file"]).strip()
    lh_path  = str(row_dict["lh_atlas"]).strip()
    rh_path  = str(row_dict["rh_atlas"]).strip()

    out_dir  = OUTPUT_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_csv = out_dir / f"{sid}_{pid}_distances.csv"
    dist_npy = out_dir / f"{sid}_{pid}_distmat.npy"
    feat_npy = out_dir / f"{sid}_{pid}_features.npy"

    # Skip if all three output files already exist (resumable runs).
    # To force recomputation, delete the relevant subject folder inside OUTPUT_DIR.
    if dist_csv.exists() and dist_npy.exists() and feat_npy.exists():
        return [
            sid, pid, "skipped_exists",
            "0.00", "", "", "", "",
            str(out_dir), str(dist_csv), str(dist_npy), str(feat_npy),
            str(EXPECTED_ROIS), "", "already computed",
        ], {
            "subject_id":     sid,
            "pet_id":         pid,
            "graph_dist_csv": str(dist_csv),
            "graph_dist_npy": str(dist_npy),
            "graph_feat_npy": str(feat_npy),
        }

    t0     = time.time()
    status = "done"
    note   = ""
    overlap_vox = ""
    t_load = t_merge = t_kde = t_dtw = 0.0
    success_row = None

    try:
        # ------------------------------------------------------------------
        # 1. Load NIfTI files
        # ------------------------------------------------------------------
        t1 = time.time()
        for p in [pet_file, lh_path, rh_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing file: {p}")

        pet_data, pet_img = load_nifti(pet_file)
        lh,       lh_img  = load_nifti(lh_path)
        rh,       rh_img  = load_nifti(rh_path)

        if pet_data.ndim != 3:
            raise RuntimeError(f"PET is not 3-D: shape={pet_data.shape}")
        if pet_data.shape != lh.shape:
            raise RuntimeError(
                f"PET/atlas shape mismatch: pet={pet_data.shape}, atlas={lh.shape}"
            )
        if not np.allclose(lh_img.affine, rh_img.affine, atol=1e-3):
            note += "WARN: lh/rh affine differ; "
        if not np.allclose(pet_img.affine, lh_img.affine, atol=1e-3):
            note += "WARN: pet/atlas affine differ; "
        if np.isfinite(pet_data).sum() == 0:
            raise RuntimeError("PET image contains no finite voxels.")
        t_load = time.time() - t1

        # ------------------------------------------------------------------
        # 2. Merge LH + RH atlases
        # ------------------------------------------------------------------
        t2 = time.time()
        atlas, astats = combine_lh_rh_atlases_safe(lh, rh, prefer=PREFER_OVERLAP)
        overlap_vox   = str(astats["overlap_vox"])
        t_merge = time.time() - t2

        # ------------------------------------------------------------------
        # 3. KDE per ROI
        # ------------------------------------------------------------------
        t3 = time.time()
        labels, values_by_label, voxel_counts = extract_region_values(
            pet_data, atlas
        )
        if min(voxel_counts.values()) < 10:
            note += f"WARN: very small ROI (min voxels={min(voxel_counts.values())}); "

        kde_by_label = {
            int(lab): kde_density(values_by_label[int(lab)], roi_label=int(lab))
            for lab in labels
        }
        t_kde = time.time() - t3

        # ------------------------------------------------------------------
        # 4. DTW pairwise distances
        # ------------------------------------------------------------------
        t4 = time.time()
        n       = len(labels)
        distmat = np.zeros((n, n), dtype=np.float64)

        for a in range(n):
            la = int(labels[a])
            for b in range(a + 1, n):
                lb   = int(labels[b])
                d    = dtw_distance(kde_by_label[la], kde_by_label[lb])
                distmat[a, b] = d
                distmat[b, a] = d

        # Sanity checks
        if not np.all(np.isfinite(distmat)):
            raise RuntimeError("Distance matrix contains NaN/Inf.")
        if not np.allclose(distmat, distmat.T, atol=1e-10):
            raise RuntimeError("Distance matrix is not symmetric.")
        if not np.allclose(np.diag(distmat), 0.0, atol=1e-12):
            raise RuntimeError("Distance matrix diagonal is not zero.")
        t_dtw = time.time() - t4

        # ------------------------------------------------------------------
        # 5. Save outputs
        # ------------------------------------------------------------------
        np.save(dist_npy, distmat)

        feat = upper_triangle_vector(distmat)
        expected_len = EXPECTED_ROIS * (EXPECTED_ROIS - 1) // 2
        if feat.shape[0] != expected_len:
            raise RuntimeError(
                f"Feature vector length {feat.shape[0]} != expected {expected_len}"
            )
        np.save(feat_npy, feat)

        with open(dist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Region 1", "Region 2", "Distance"])
            for a in range(n):
                for b in range(a + 1, n):
                    w.writerow([int(labels[a]), int(labels[b]), float(distmat[a, b])])

        success_row = {
            "subject_id":     sid,
            "pet_id":         pid,
            "graph_dist_csv": str(dist_csv),
            "graph_dist_npy": str(dist_npy),
            "graph_feat_npy": str(feat_npy),
        }

    except Exception as e:
        status = "failed"
        note   = f"{type(e).__name__}: {e}"

    total_sec = time.time() - t0
    log_row = [
        sid, pid, status,
        f"{total_sec:.2f}", f"{t_load:.2f}", f"{t_merge:.2f}",
        f"{t_kde:.2f}", f"{t_dtw:.2f}",
        str(out_dir), str(dist_csv), str(dist_npy), str(feat_npy),
        str(EXPECTED_ROIS), overlap_vox, note,
    ]
    return log_row, success_row


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # Validate configuration
    if str(COHORT_CSV) == "*** INSERT PATH TO YOUR COHORT CSV ***":
        raise ValueError(
            "Please set COHORT_CSV in the CONFIGURATION section before running."
        )
    if str(OUTPUT_DIR) == "*** INSERT PATH TO YOUR OUTPUT DIRECTORY ***":
        raise ValueError(
            "Please set OUTPUT_DIR in the CONFIGURATION section before running."
        )
    if not COHORT_CSV.exists():
        raise FileNotFoundError(f"Cohort CSV not found: {COHORT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir      = OUTPUT_DIR / "logs"
    progress_csv = log_dir / "progress.csv"
    master_csv   = OUTPUT_DIR / "graphs_master.csv"
    init_progress_log(progress_csv)

    df = pd.read_csv(COHORT_CSV, low_memory=False)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Cohort CSV missing required columns: {missing_cols}")

    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str).str.strip()

    dup = df.duplicated(subset=["subject_id", "pet_id"], keep=False)
    if dup.any():
        raise ValueError(
            f"Duplicate (subject_id, pet_id) rows found in cohort CSV:\n"
            f"{df.loc[dup, ['subject_id','pet_id']].head(10).to_string(index=False)}"
        )

    rows  = df.to_dict(orient="records")
    total = len(rows)

    info("=" * 70)
    info("METABOLIC BRAIN GRAPH CONSTRUCTION")
    info("=" * 70)
    info(f"Cohort CSV    : {COHORT_CSV}")
    info(f"Output dir    : {OUTPUT_DIR}")
    info(f"Subjects      : {total}")
    info(f"CPUs detected : {TOTAL_CPUS}")
    info(f"CPU fraction  : {CPU_FRACTION:.2f}  →  MAX_WORKERS = {MAX_WORKERS}")
    info(f"KDE grid size : {KDE_GRID_N}  (2^{int(np.log2(KDE_GRID_N))})")
    info(f"ROIs          : {EXPECTED_ROIS}")

    t_all        = time.time()
    times_done   = []
    success_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(process_one_graph, row): i
                      for i, row in enumerate(rows)}

        for future in as_completed(future_map):
            i   = future_map[future]
            row = rows[i]
            sid = str(row["subject_id"]).strip()
            pid = str(row["pet_id"]).strip()

            try:
                log_row, success_row = future.result()
            except Exception as e:
                log_row = [
                    sid, pid, "failed", "", "", "", "", "",
                    "", "", "", "", "", "",
                    f"UnhandledFutureException: {type(e).__name__}: {e}",
                ]
                success_row = None

            append_log(progress_csv, log_row)

            if log_row[2] == "done":
                try:
                    times_done.append(float(log_row[3]))
                except Exception:
                    pass

            if success_row is not None:
                success_rows.append(success_row)

            print(f"[{i+1}/{total}] {sid} {pid} -> {log_row[2]}", flush=True)

    elapsed = time.time() - t_all

    # ------------------------------------------------------------------
    # Write master CSV of all successful graphs
    # ------------------------------------------------------------------
    if success_rows:
        master_df = pd.DataFrame(success_rows)
        master_df = master_df.sort_values(
            ["subject_id", "pet_id"], kind="mergesort"
        ).reset_index(drop=True)
        master_df.to_csv(master_csv, index=False)
        info(f"Master CSV written : {master_csv}  ({len(master_df)} rows)")
    else:
        warn("No successful graphs — master CSV not written.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_done   = len(success_rows)
    n_failed = total - n_done

    info("=" * 70)
    info("SUMMARY")
    info("=" * 70)
    info(f"Total subjects   : {total}")
    info(f"Successful       : {n_done}")
    info(f"Failed           : {n_failed}")
    info(f"Total wall time  : {elapsed/60:.1f} min")
    info(f"Progress log     : {progress_csv}")

    if times_done:
        info(f"Per-subject time (DONE cases, parallel wall clock):")
        info(f"  mean   : {float(np.mean(times_done)):.1f} s")
        info(f"  median : {float(np.median(times_done)):.1f} s")
        info(f"  min    : {float(np.min(times_done)):.1f} s")
        info(f"  max    : {float(np.max(times_done)):.1f} s")

    if n_failed > 0:
        info(f"Check {progress_csv} for failure reasons.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise