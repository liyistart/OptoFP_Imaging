# OptoFP 2P Imaging Analysis and Modeling

This repository contains code used to analyze 2‑photon calcium imaging data during a mouse reach‑to‑consume task with and without 
optogenetic muscle stimulation. `batch2Ana_Fezf2_Population_Final_20260204.ipynb` used to generate population‑level data analysis figures. 

This repository also contains analysis code and notebooks for fitting and evaluating a neural–behavior coupling model for the reach-to-consume task. The notebooks reproduce the main computational modeling results and figures in the paper.  
`all_session_summary_plot.ipynb`, `session4_plot.ipynb`, and `no_pertubation_test.ipynb` generate figures related to the computational model.


## Notebook
- `batch2Ana_Fezf2_Population_Final_20260204.ipynb`
  - Population analysis for Fezf2 ET neurons during a reach‑to‑consume task with/without optogenetic muscle stimulation.

- `session4_plot.ipynb`
  - Single-session computational model fitting and reproduction of figures used in the paper.

- `all_session_summary_plot.ipynb`
  - Multi-session computational model analysis. This notebook aggregates results across all sessions and generates summary figures used for population-level evaluation in the paper.

- `no_pertubation_test.ipynb`
  - Model comparison across variants, including: 
  
      - Intrinsic-only dynamics (no external kinematic input);
      - Intrinsic + unperturbed kinematic feedback;
      - Full model with perturbation-dependent feedback gating.  

## Requirements
Python packages used directly in the notebook:
- `numpy`
- `scipy`
- `pandas`
- `pickle`
- `matplotlib`
- `scikit-learn`
- `pingouin`

Local modules required (must be importable from the repo root):
- `AdvancedAnalysis_2P.py`
- `BasicProcessing_2P.py`
- `regression_utils.py` 

## Data Files (required)
For general data analysis, please place these in the repo root (same folder as the notebook), unless you update paths inside the notebook:
- `Fezf2_OptoFP_2P_Spks_Sum.pkl`
- `Fezf2_Sortinfo.pkl`

The data files for running computational modeling has been put in the directory `/modeling_data`, these data is directly extracted from `Fezf2_OptoFP_2P_Spks_Sum.pkl` with no other preprocessing. 


## Installation
```
conda create -n optoFP python=3.11
conda activate optoFP
pip install -r requirements.txt
```

## Run
```bash
jupyter lab
```
Open the desired `.ipynb` file and run all cells.

## Outputs
The notebook `batch2Ana_Fezf2_Population_Final_20260204.ipynb` writes PDF figures to the working directory, including:
- `Fezf2_Session_LDA_Blocks_Lineplot_Summary_Spks.pdf`
- `All_Fezf2_PCA_Neural_Trajectory_Spk_activity_2D_All_Neurons.pdf`
- `All_Fezf2_PCA_Neural_Trajectory_Spk_activity_3D_All_Neurons.pdf`
- `Fezf2_PCA_Neural_Trajectory_Spk_activity_Off_Plane_Proj_All_Neurons.pdf`
- `Fezf2_PCA_Neural_Trajectory_Spk_activity_Off_Plane_Vel_All_Neurons.pdf`

## Citation
If you use this dataset, please cite the associated manuscript or preprint:

Distorted Proprioceptive Feedback Reshapes Descending Cortical Dynamics  
Yi Li, X. Hermione Xu, Shiyang Pan, Xu An, Patrick J. Mulcahey, Zixuan Qiu, Shengli Zhao, Nuo Li, John Pearson, Z. Josh Huang  
bioRxiv 2025.11.11.687832; doi: 10.1101/2025.11.11.687832

## Contact
Please contact Dr. Yi Li (liyi.start@gmail.com) for questions about the dataset.
