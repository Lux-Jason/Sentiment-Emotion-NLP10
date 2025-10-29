# Pseudo-Civility Detection Experiment Report

## Experiment: enhanced_binary_civility_detection
**Duration:** 0.00 seconds
**Status:** Failed

## Configuration

- **Datasets:** chnsenticorp, wikipedia_politeness, go_emotions, civil_comments, toxigen
- **Classifier:** random_forest
- **Embedding Model:** Qwen/Qwen3-Embedding-4B
- **Device:** auto
- **Batch Size:** 4
- **Max Samples:** 400

## Data Statistics

- **Total Samples:** 744
- **Training:** 517
- **Validation:** 76
- **Test:** 151

## Per-Dataset Results

### Dataset: wikipedia_politeness

- **Samples:** train 346, val 50, test 100
- **Artifacts:** `outputs\enhanced_binary_civility_detection\run_005\wikipedia_politeness`

- **Validation Metrics:**
  - accuracy: 0.8400
  - f1: 0.8519
  - precision: 0.7931
  - recall: 0.9200
- **Test Metrics:**
  - accuracy: 0.8400
  - f1: 0.8462
  - precision: 0.8148
  - recall: 0.8800
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9596
  - pr_auc: 0.9637

### Dataset: civil_comments

- **Samples:** train 66, val 10, test 20
- **Artifacts:** `outputs\enhanced_binary_civility_detection\run_005\civil_comments`

- **Validation Metrics:**
  - accuracy: 0.9000
  - f1: 0.8889
  - precision: 1.0000
  - recall: 0.8000
- **Test Metrics:**
  - accuracy: 0.8500
  - f1: 0.8571
  - precision: 0.8182
  - recall: 0.9000
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9500
  - pr_auc: 0.9466

### Dataset: toxigen

- **Samples:** train 105, val 16, test 31
- **Artifacts:** `outputs\enhanced_binary_civility_detection\run_005\toxigen`

- **Validation Metrics:**
  - accuracy: 0.8125
  - f1: 0.8000
  - precision: 0.8571
  - recall: 0.7500
- **Test Metrics:**
  - accuracy: 0.7097
  - f1: 0.7097
  - precision: 0.6875
  - recall: 0.7333
- **Test Probabilistic Metrics:**
  - roc_auc: 0.7875
  - pr_auc: 0.8159

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
