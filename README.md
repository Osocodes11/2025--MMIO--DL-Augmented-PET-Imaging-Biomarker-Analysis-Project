# 2025--MMIO--DL-Augmented-PET-Imaging-Biomarker-Analysis-Project
Overview
This repository contains Python scripts used to analyze PET imaging biomarkers derived from both conventional and deep learning–augmented PET images. The project focused on extracting regional imaging features, preparing machine learning–ready datasets, and evaluating classification models for disease stratification. The scripts demonstrate workflows for dataset restructuring, supervised machine learning experiments, and feature importance analysis.

Project Objectives
- Extract regional PET imaging features from brain regions of interest (ROI)
- Structure multimodal imaging data into machine learning–ready datasets
- Train and evaluate classification models
- Analyze feature importance to investigate imaging biomarkers

Key Components
1. ROI Feature Extraction
Scripts were used to extract regional PET imaging features from predefined brain regions. Features included:
- SUVR and non-SUVR-based measurements
- Parametric PET maps
- Regional activity values were computed for each patient and organized into structured datasets.

2. Dataset Restructuring
Imaging feature tables were transformed from long format to wide format, producing patient-level feature matrices suitable for machine learning analysis.

3. Machine Learning Evaluation
Classification models were trained using XGBoost with stratified 5-fold cross-validation. Model evaluation included:
- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
Confusion matrices and classification reports were generated for each fold.

4. Feature Importance Analysis
Feature importance scores were aggregated across folds to identify regional imaging features that contributed most strongly to classification performance.

Tools & Technologies
- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
