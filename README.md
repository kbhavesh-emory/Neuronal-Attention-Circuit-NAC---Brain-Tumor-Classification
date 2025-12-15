# Neuronal Attention Circuit (NAC) — Brain Tumor Classification

<div align="center">
	<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
	<img src="https://img.shields.io/badge/Medical%20Imaging-00BFFF?style=for-the-badge" alt="Medical Imaging" />
	<img src="https://img.shields.io/badge/Vision%20Transformer-FF6B6B?style=for-the-badge" alt="Vision Transformer" />
	<img src="https://img.shields.io/badge/Bio--Inspired-8A2BE2?style=for-the-badge" alt="Bio-Inspired" />
</div>

Biologically inspired continuous-time attention (NAC) integrated into a ViT-style image classifier for brain tumor MRI classification.

## Contents
- [Overview](#overview)
- [Method (NAC)](#method-nac)
- [Model](#model)
- [Dataset](#dataset)
- [Run the notebook](#run-the-notebook)
- [Results](#results)
- [Project structure](#project-structure)
- [Notes](#notes)
- [Citation](#citation)

## Overview
This repository contains a single end-to-end Jupyter notebook that:
- loads and augments a 4-class brain tumor MRI dataset,
- implements Neuronal Attention Circuit (NAC) attention (continuous-time ODE-inspired attention logits),
- builds a Hybrid NAC–ViT architecture (ViT patch embeddings + NAC blocks),
- trains/evaluates the model and visualizes metrics and attention behavior.

## Method (NAC)
The NAC attention module is designed to be:
- Continuous-time: attention logits derived from a closed-form ODE solution.
- Sparse: Top-K key selection per query for efficiency.
- Stable: clamped learnable pseudo-time parameter and logits.

## Model
The notebook defines a Hybrid NAC–ViT classifier:
- backbone: torchvision ViT-B/16 patch embedding + positional embeddings
- NAC blocks: 4 stacked blocks (LayerNorm → NAC Attention → MLP)
- head: MLP classifier to 4 classes

## Dataset
The notebook is written for the Kaggle Brain Tumor MRI Dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected folder layout (after you download and extract):
```
data/
	brain_tumor_mri/
		glioma/
		meningioma/
		notumor/
		pituitary/
```

## Run the notebook
1) Install dependencies (example):
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm
```

2) Open and run:
- Brain-Tumor-Classification.ipynb

3) Update the dataset path inside the notebook:
- DatasetConfig.DATA_DIR = "./data/brain_tumor_mri"

## Results
The notebook includes evaluation plots (confusion matrix, ROC/PR curves, calibration, confidence distributions) and a baseline comparison vs a standard ViT.

Important: if the dataset path is missing, the notebook may generate dummy images to smoke-test the pipeline. In that case, metrics/plots are not meaningful (you’ll see noise-like images and low accuracy). Download the real dataset and point DATA_DIR at it for real results.

## Project structure
```
.
├── Brain-Tumor-Classification.ipynb
└── README.md
```

## Notes
- This project is for research/education and is not a medical device.
- Reproducibility: the notebook sets random seeds, but GPU nondeterminism can still affect runs.

## Citation
If you use this work, please cite this repository (or add your paper/preprint citation here).

 