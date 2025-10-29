# Pseudo-Civility Detection Experiment Report

## Experiment: binary_civility_detection
**Duration:** 0.00 seconds
**Status:** Failed

## Configuration

- **Datasets:** chnsenticorp, wikipedia_politeness, go_emotions, civil_comments, toxigen
- **Classifier:** logistic_regression
- **Embedding Model:** Qwen/Qwen3-Embedding-4B
- **Device:** auto
- **Batch Size:** 8
- **Max Samples:** 400

## Data Statistics

- **Total Samples:** 1778
- **Training:** 1241
- **Validation:** 179
- **Test:** 358

## Per-Dataset Results

### Dataset: chnsenticorp

- **Samples:** train 283, val 41, test 82
- **Artifacts:** `outputs\binary_civility_detection\run_007\chnsenticorp`

- **Validation Metrics:**
  - accuracy: 0.8780
  - f1: 0.8718
  - precision: 0.8947
  - recall: 0.8500
- **Test Metrics:**
  - accuracy: 0.9024
  - f1: 0.9000
  - precision: 0.9231
  - recall: 0.8780
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9399
  - pr_auc: 0.9552

### Dataset: wikipedia_politeness

- **Samples:** train 350, val 50, test 100
- **Artifacts:** `outputs\binary_civility_detection\run_007\wikipedia_politeness`

- **Validation Metrics:**
  - accuracy: 0.7200
  - f1: 0.7500
  - precision: 0.6774
  - recall: 0.8400
- **Test Metrics:**
  - accuracy: 0.7200
  - f1: 0.7358
  - precision: 0.6964
  - recall: 0.7800
- **Test Probabilistic Metrics:**
  - roc_auc: 0.7976
  - pr_auc: 0.7947

### Dataset: civil_comments

- **Samples:** train 66, val 10, test 20
- **Artifacts:** `outputs\binary_civility_detection\run_007\civil_comments`

- **Validation Metrics:**
  - accuracy: 0.8000
  - f1: 0.8000
  - precision: 0.8000
  - recall: 0.8000
- **Test Metrics:**
  - accuracy: 0.8500
  - f1: 0.8421
  - precision: 0.8889
  - recall: 0.8000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9600
  - pr_auc: 0.9635

### Dataset: toxigen

- **Samples:** train 262, val 38, test 76
- **Artifacts:** `outputs\binary_civility_detection\run_007\toxigen`

- **Validation Metrics:**
  - accuracy: 0.7105
  - f1: 0.6857
  - precision: 0.7500
  - recall: 0.6316
- **Test Metrics:**
  - accuracy: 0.8421
  - f1: 0.8333
  - precision: 0.8824
  - recall: 0.7895
- **Test Probabilistic Metrics:**
  - roc_auc: 0.8920
  - pr_auc: 0.8370

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
