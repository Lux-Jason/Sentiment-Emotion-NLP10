# Pseudo-Civility Detection Experiment Report

## Experiment: enhanced_binary_civility_detection
**Duration:** 0.00 seconds
**Status:** Failed

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
- **Artifacts:** `Codes\QLoRa\outputs\enhanced_binary_civility_detection\run_007\chnsenticorp`

- **Validation Metrics:**
  - accuracy: 0.9006
  - f1: 0.8967
  - precision: 0.9328
  - recall: 0.8632
- **Test Metrics:**
  - accuracy: 0.8909
  - f1: 0.8867
  - precision: 0.9215
  - recall: 0.8545
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9363
  - pr_auc: 0.9450

### Dataset: wikipedia_politeness

- **Samples:** train 3024, val 432, test 864
- **Artifacts:** `Codes\QLoRa\outputs\enhanced_binary_civility_detection\run_007\wikipedia_politeness`

- **Validation Metrics:**
  - accuracy: 0.7523
  - f1: 0.7397
  - precision: 0.7795
  - recall: 0.7037
- **Test Metrics:**
  - accuracy: 0.7234
  - f1: 0.7110
  - precision: 0.7443
  - recall: 0.6806
- **Test Probabilistic Metrics:**
  - roc_auc: 0.8012
  - pr_auc: 0.8057

### Dataset: go_emotions

- **Samples:** train 6503, val 930, test 1859
- **Artifacts:** `Codes\QLoRa\outputs\enhanced_binary_civility_detection\run_007\go_emotions`

- **Validation Metrics:**
  - accuracy: 0.8806
  - f1: 0.8743
  - precision: 0.9234
  - recall: 0.8301
- **Test Metrics:**
  - accuracy: 0.8779
  - f1: 0.8730
  - precision: 0.9091
  - recall: 0.8396
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9353
  - pr_auc: 0.9457

### Dataset: civil_comments

- **Samples:** train 2303, val 330, test 659
- **Artifacts:** `Codes\QLoRa\outputs\enhanced_binary_civility_detection\run_007\civil_comments`

- **Validation Metrics:**
  - accuracy: 0.8606
  - f1: 0.8562
  - precision: 0.8839
  - recall: 0.8303
- **Test Metrics:**
  - accuracy: 0.8467
  - f1: 0.8453
  - precision: 0.8519
  - recall: 0.8389
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9187
  - pr_auc: 0.9122

### Dataset: toxigen

- **Samples:** train 992, val 142, test 284
- **Artifacts:** `Codes\QLoRa\outputs\enhanced_binary_civility_detection\run_007\toxigen`

- **Validation Metrics:**
  - accuracy: 0.8803
  - f1: 0.8811
  - precision: 0.8750
  - recall: 0.8873
- **Test Metrics:**
  - accuracy: 0.7711
  - f1: 0.7687
  - precision: 0.7770
  - recall: 0.7606
- **Test Probabilistic Metrics:**
  - roc_auc: 0.8434
  - pr_auc: 0.8040

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
