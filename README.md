# Breast Tumor Classification: Model Performance, Validation and Robustness

A machine-learning study of breast tumor classification using the Wisconsin Breast Cancer Original dataset, with a focus on validation, robustness, probability calibration, feature stability, and prediction uncertainty.

## Research Question

How does evaluation methodology affect the apparent reliability of machine-learning models for breast-tumor classification?

## Research Questions

This study investigates:

1. How consistently do classical machine-learning models perform under repeated stratified validation?
2. Does increasing neural-network complexity produce meaningful improvements on a small tabular dataset?
3. How do classification thresholds affect sensitivity and specificity?
4. How well calibrated are model probabilities?
5. Are difficult cases consistently misclassified across different model families?
6. Are feature importance estimates stable across validation folds?
7. How sensitive are results to the treatment of missing values?
8. Can confidence-based abstention reduce errors among accepted predictions?
9. How does a modern tabular model such as TabPFN compare with conventional approaches?

## Dataset

The primary dataset is the **Wisconsin Breast Cancer Original (WBC)** dataset from the UCI Machine Learning Repository.

- 699 observations
- 9 integer-valued features
- 458 benign observations
- 241 malignant observations
- Original target labels:
  - `2` = benign
  - `4` = malignant
- 16 observations contain missing values in `Bare_nuclei`

The nine features describe characteristics of cell nuclei recorded from breast-tumor samples.

The UCI documentation also describes chronological collection groups. However, the corresponding group labels are not included in the returned feature table. Therefore, this study does not reconstruct or assume chronological ordering.

## Methodology

The analysis follows a reproducible machine-learning workflow:

```text
Dataset inspection
        ↓
Missing-value analysis
        ↓
Stratified train/test split
        ↓
Preprocessing pipelines
        ↓
Classical baseline models
        ↓
Repeated stratified cross-validation
        ↓
Threshold analysis
        ↓
Probability calibration
        ↓
Neural-network complexity analysis
        ↓
Error and uncertainty analysis
        ↓
TabPFN benchmark
        ↓
Feature stability analysis
        ↓
Missing-data sensitivity analysis
        ↓
Confidence-based abstention
        ↓
Final interpretation