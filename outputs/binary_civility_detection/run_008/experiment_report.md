# Pseudo-Civility Detection Experiment Report

## Experiment: binary_civility_detection
**Duration:** 0.00 seconds
**Status:** Failed

## Configuration

- **Datasets:** go_emotions
- **Classifier:** logistic_regression
- **Embedding Model:** Qwen/Qwen3-Embedding-4B
- **Device:** auto
- **Batch Size:** 16
- **Max Samples:** All

## Data Statistics

- **Total Samples:** 9292
- **Training:** 6503
- **Validation:** 930
- **Test:** 1859

## Per-Dataset Results

### Dataset: go_emotions

- **Samples:** train 6503, val 930, test 1859
- **Artifacts:** `outputs\binary_civility_detection\run_008\go_emotions`

- **Validation Metrics:**
  - accuracy: 0.8806
  - f1: 0.8746
  - precision: 0.9214
  - recall: 0.8323
- **Test Metrics:**
  - accuracy: 0.8752
  - f1: 0.8697
  - precision: 0.9095
  - recall: 0.8332
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9392
  - pr_auc: 0.9451

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
