# XGML

Code supporting the manuscript:

**Explainable Graph-theoretical Machine Learning with Application to Alzheimer's Disease Prediction**

## Overview

This repository contains code for constructing individual metabolic brain graphs from FDG-PET scans and for applying an explainable graph-theoretical machine learning (XGML) framework to cognitive score prediction.

The graph-construction pipeline is based on:

- kernel density estimation (KDE) of PET voxel intensities within each brain region
- dynamic time warping (DTW) to quantify pairwise distances between regional intensity distributions
- the Schaefer 2018 200-parcel atlas

The prediction pipeline supports:

- internal validation by repeated stratified cross-validation
- external validation across datasets
- permutation feature importance analysis

## Repository contents

- `adni_graph_construction.py`  
  Constructs subject-level metabolic brain graphs from ADNI FDG-PET data.

- `oasis_graph_construction.py`  
  Constructs subject-level metabolic brain graphs from OASIS3 FDG-PET data. The same procedure can also be adapted to other datasets containing FDG-PET scans.

- `predictions_most_pred_features.py`  
  Runs the prediction pipeline, including internal validation, external validation, and feature-importance analysis.

- `requirements.txt`  
  Python dependencies required to run the code.

## Data availability

This repository does not include ADNI or OASIS3 data.

Researchers must obtain access to the datasets separately and comply with their respective data use conditions.

- **ADNI**: access via the Alzheimer's Disease Neuroimaging Initiative
- **OASIS3**: access via the OASIS project

## Input requirements

### Graph construction scripts

The input cohort CSV must contain at least:

- `subject_id`
- `pet_id`
- `pet_file`
- `lh_atlas`
- `rh_atlas`

where:

- `pet_file` is the path to the preprocessed FDG-PET NIfTI file
- `lh_atlas` is the path to the left-hemisphere atlas NIfTI
- `rh_atlas` is the path to the right-hemisphere atlas NIfTI

### Prediction pipeline

The input CSV must contain at least:

- `subject_id`
- `pet_id`
- `group` (required for internal validation)
- one or more target cognitive score columns

Feature files are expected as NumPy arrays saved under:

```text
<GRAPH_ROOT>/<subject_id>/<subject_id>_<pet_id>_features.npy
