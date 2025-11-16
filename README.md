# HE2FISH

## Data-efficient deep learning for gene rearrangement status prediction in DLBCL
*Pending* <img src="Images/ICON" width="250px" align="right" />

We developed HE2FISH, a deep learning-based framework for predicting gene rearrangements in Diffuse Large B-Cell Lymphoma (DLBCL) directly from whole-slide pathology images. As illustrated in the overview, HE2FISH comprises three main stages: data pre-processing, multi-scale instance aggregation, and interpretable marker prediction.

<div style="text-align:center">
  <img src="Images/Method.png" width="800px" />
</div>

## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/MedCAI/HE2FISH.git
cd HE2FISH
```
Then create a conda env and install the dependencies:
```shell
conda env create -f environment.yml
conda activate he2fish
```
## Feature Extraction
Our feature extractor is UNI-h2 (https://github.com/mahmoodlab/uni). Specifically, we pre-process each whole-slide image using Trident (https://github.com/mahmoodlab/TRIDENT) and obtain the patch features at both 5x and 20x. T
