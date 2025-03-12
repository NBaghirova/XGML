# Explainable Graph-theoretical Machine Learning (XGML)

A Python repository for Explainable Graph-Theoretical Machine Learning (XGML), a framework, which employs kernel density estimation and dynamic time warping to construct individual brain graphs from neuroimaging data that capture the distance between pair-wise brain regions. Additionally, XGML identifies subgraphs predictive of multivariate disease-related outcomes. To illustrate its efficacy, we use XGML to construct the graphs from FDG-PET scans from the Alzheimer's Disease Neuroimaging Initiative and uncover subgraphs predictive of eight AD-related cognitive scores in previously unseen subjects.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Metabolic Graph Construction and Predictions](#metabolic-graph-construction-and-predictions)
3. [Outputs](#outputs) 
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Directory Structure](#directory-structure)
7. [Usage](#usage)
8. [References](#references)

---

## Introduction

We apply XGML to Alzheimer’s Disease (AD) study. AD is a neurodegenerative disorder and the most common form of dementia. It severely impacts memory, cognition, and daily functioning. According to Alzheimer’s Research UK 2021, by 2050 the number of people affected by dementia is projected to reach 152 million globally with AD accounting for most cases. Early detection and accurate severity assessment are critical, as current treatments are most effective during the disease’s early stages.

In this Python repository, we provide scripts that leverage the XGML framework to construct metabolic distance graphs from FDG-PET scans in the Alzheimer's Disease Neuroimaging Initiative dataset. These scripts predict eight AD-related cognitive scores and identify subgraphs that are predictive of these scores in previously unseen subjects.

---

## Metabolic Graph Construction and Predictions

1. The *constructing_metabolic_similarity_graphs.py* script constructs metabolic similarity graphs from FDG-PET scans for each patient.

First, the script scans a directory to locate FDG-PET images that match a specified file suffix. Then, it loads a brain atlas, more specifically, we utilized a pre-processed Schaefer 2018 atlas with 200 brain regions, which is dimensionally aligned with the pre-processed FDG-PET scans. For each identified by the atlas brain region, the script extracts intensity values from the PET images. Further, Kernel Density Estimation (KDE) is applied to model the probability distribution of intensity values within each brain region. Then, Dynamic Time Warping (DTW) algorithm is applied to compute the similarity between KDE distributions of all pairs of brain regions. Thus, for each patient and their corresponding FDG-PET scan, the metabolic similarity graph is constructed as follows. The vertices of a graph represent brain regions. The graph forms a clique (every pair of vertices is adjacent). Edges are weighted by the DTW similarity scores between corresponding regions. Finally, the computed similarity values are stored in a CSV file for further analysis.

2. The *KernelSVR_top_10_edges.py* script applies the Kernel SVR model to predict cognitive scores and identifies subgraphs that are predictive of each of the cognitive scores in previously unseen subjects.

First, the script loads the metabolic similarity graphs and cognitive scores, ensuring consistency in subject data. Each similarity graph is vectorized by extracting the upper triangular values, forming the feature set, while cognitive scores serve as target variables.

Using the best Kernel SVR model with the following hyperparameters: kernel='rbf', C=10, gamma=0.001, degree=2, epsilon=0.2, the script trains and evaluates the model through Leave-One-Out Cross-Validation (LOOCV) to generate final cognitive score predictions. The predicted vs. actual values are then stored in CSV files.

To determine the most important edges in predicting cognitive scores, permutation importance is computed for each cognitive score. The script identifies the top 10 most predictive edges in the similarity graph by ranking feature importance scores. These edges are then mapped back to their corresponding brain regions.

Finally, for each cognitive score, a 200 × 200 adjacency matrix is generated, where the most predictive edges are marked, and the results are stored in CSV format for further analysis.

---

## Outputs

The outputs repository includes two folders:

1. top_10_predictive_edges/ – Contains eight CSV files, each containing the top 10 most predictive edges for one of the cognitive scores. These edges represent the most important connections identified by the model.
2. true_vs_predicted/ – Contains eight CSV files, each storing the true vs. predicted cognitive scores for all subjects. These files provide values of the model’s predictions and actual cognitive scores.

All output files were generated using the scripts provided and the dataset used in the study.

---

## Requirements

To run the scripts in this repository, ensure you have the following dependencies installed:

**Python Version**

Python 3.7 or later

**Required Libraries**

You can install the required Python libraries using the following command:

```
pip install numpy pandas nibabel scikit-learn KDEpy dtw
```

Alternatively, you can install them manually:

- NumPy – For numerical operations (pip install numpy)
- Pandas – For data handling and manipulation (pip install pandas)
- Nibabel – To work with neuroimaging data (pip install nibabel)
- Scikit-learn – For machine learning algorithms and model evaluation (pip install scikit-learn)
- KDEpy – For Kernel Density Estimation (pip install KDEpy)
- dtw – For Dynamic Time Warping similarity computation (pip install dtw)

**Data Requirements**

Preprocessed FDG-PET scans in NIfTI (.nii.gz) format.
A brain atlas (e.g., Schaefer 2018 atlas) that aligns with the FDG-PET scans.
A CSV file containing subject IDs and corresponding cognitive scores.
A directory containing precomputed metabolic similarity graphs (for training the Kernel SVR model).

---
## Installation

Follow these steps to set up and run the repository on your computer:

**1. Clone the Repository**

First, clone this GitHub repository to your local system.

**2. Install Dependencies**

Install the required packages by running:

```
pip install -r requirements.txt
```
**3. Set Up the Data**

Ensure the following files are available in the appropriate directories:

- FDG-PET scans in NIfTI (.nii.gz) format
- A brain atlas file (e.g., Schaefer 2018 atlas)
- A CSV file containing subject IDs and cognitive scores
- A directory with precomputed metabolic similarity graphs (if running SVR training)
- Update the paths in the scripts (constructing_metabolic_similarity_graphs.py, and KernelSVR_BP_top_10_edges.py) to match your dataset locations.

**4. Run the Scripts**

Finally, you can run each script as follows:

```
python constructing_metabolic_similarity_graphs.py
python KernelSVR_BP_top_10_edges.py
```

Each script will generate output files in the designated directories.

---
## Directory Structure

The project directory is organized as follows:

```
Explainable-Graph-theoretical-Machine-Learning/
├── scripts/
│   ├── constructing_metabolic_similarity_graphs.py       
│   ├── KernelSVR_BP_top_10_edges.py
├── outputs/
│   ├── top 10 predictive edges/
│   │   ├──ADAS11_top_10_edges.csv
│   │   ├──ADAS13_top_10_edges.csv
│   │   ├──ADASQ4_top_10_edges.csv
│   │   ├──CDRSB_top_10_edges.csv
│   │   ├──MMSE_top_10_edges.csv
│   │   ├──RAVLT_immediate_top_10_edges.csv
│   │   ├──RAVLT_learning_top_10_edges.csv
│   │   ├──RAVLT_perc_forgetting_top_10_edges.csv
│   ├── true vs. predicted/ 
│   │   ├──ADAS11_ytrue_ypred.csv
│   │   ├──ADAS13_ytrue_ypred.csv
│   │   ├──ADASQ4_ytrue_ypred.csv
│   │   ├──CDRSB_ytrue_ypred.csv
│   │   ├──MMSE_ytrue_ypred.csv
│   │   ├──RAVLT_immediate_ytrue_ypred.csv
│   │   ├──RAVLT_learning_ytrue_ypred.csv
│   │   ├──RAVLT_perc_forgetting_ytrue_ypred.csv
├── README.md         
└── requirements.txt
```
---
## Usage

Here, we outline how to run the scripts in the correct order and interpret the outputs. Before running the scripts, ensure that all file paths in the scripts are properly updated to reflect your dataset locations and that all required data and files are accessible.

**1. Construct Metabolic Similarity Graphs**

Run the *constructing_metabolic_similarity_graphs.py* script to generate metabolic similarity graphs from FDG-PET scans.

```
python constructing_metabolic_similarity_graphs.py 
```

Ensure that the paths to the FDG-PET scans and brain atlas are correctly set in the script. Additionally, ensure that the scans are dimensionally aligned with the atlas.

Expected Output: CSV files containing similarity matrices for each patient, stored in the specified directory.


**2. Identify the Top 10 Most Predictive Edges**

Run the *KernelSVR_top_10_edges.py* script to determine the most important edges in the metabolic similarity graphs for predicting each cognitive score. This script applies permutation importance to identify the top 10 most predictive edges for each cognitive score.

```
python KernelSVR_BP_top_10_edges.py
```

Ensure that the paths to the trained model, similarity graph files, and cognitive scores CSV are correctly updated in the script.

Expected Output:
- Eight adjacency matrices of size 200 × 200, containing the most predictive edges (the corresponding entry in the matrix equals 1 if the edge is predictive; 0 otherwise).
- CSV files storing those matrices.

---
## References
1. Alzheimer’s Research UK. Worldwide dementia cases to triple by 2050, 2021. 
2. M. Rosselli, I. V. Uribe, E. Ahne, and L. Shihadeh. Culture, ethnicity, and level of edu-
cation in alzheimer’s disease. Neurotherapeutics: The Journal of the American Society for
Experimental NeuroTherapeutics, 19(1):26–54, 2022.
