# Pseudo-Civility Detection Experiment Report

## Experiment: binary_civility_detection
**Duration:** 0.00 seconds
**Status:** ❌ Failed

## Configuration

- **Datasets:** chnsenticorp, wikipedia_politeness, go_emotions, civil_comments, toxigen
- **Classifier:** logistic_regression
- **Embedding Model:** Qwen/Qwen3-Embedding-4B
- **Device:** auto
- **Batch Size:** 16
- **Max Samples:** All

## Data Statistics

- **Total Samples:** 29590
- **Training:** 20709
- **Validation:** 2961
- **Test:** 5920

## Per-Dataset Results

### Dataset: chnsenticorp

- **Samples:** train 7887, val 1127, test 2254
- **Artifacts:** `outputs\binary_civility_detection\run_004\chnsenticorp`

- **Validation Metrics:**
  - accuracy: 0.5004
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.5000
  - pr_auc: 0.7500

### Dataset: wikipedia_politeness

- **Samples:** train 3024, val 432, test 864
- **Artifacts:** `outputs\binary_civility_detection\run_004\wikipedia_politeness`

- **Validation Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.5000
  - pr_auc: 0.7500

### Dataset: go_emotions

- **Samples:** train 6503, val 930, test 1859
- **Artifacts:** `outputs\binary_civility_detection\run_004\go_emotions`

- **Validation Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Metrics:**
  - accuracy: 0.5003
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.5000
  - pr_auc: 0.7499

### Dataset: civil_comments

- **Samples:** train 2303, val 330, test 659
- **Artifacts:** `outputs\binary_civility_detection\run_004\civil_comments`

- **Validation Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Metrics:**
  - accuracy: 0.5008
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.5000
  - pr_auc: 0.7496

### Dataset: toxigen

- **Samples:** train 992, val 142, test 284
- **Artifacts:** `outputs\binary_civility_detection\run_004\toxigen`

- **Validation Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Metrics:**
  - accuracy: 0.5000
  - f1: 0.0000
  - precision: 0.0000
  - recall: 0.0000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.5000
  - pr_auc: 0.7500

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
