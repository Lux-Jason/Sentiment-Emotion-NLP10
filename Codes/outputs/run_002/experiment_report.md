# Pseudo-Civility Detection Experiment Report

## Experiment: full_run_002
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
- **Artifacts:** `d:\E_textbooks_And_Lecture_Notes\NLP\NLP10\Codes\QLoRa\outputs\full_run_002\run_002\chnsenticorp`

- **Validation Metrics:**
  - accuracy: 0.9042
  - f1: 0.9011
  - precision: 0.9301
  - recall: 0.8739
- **Test Metrics:**
  - accuracy: 0.8873
  - f1: 0.8843
  - precision: 0.9083
  - recall: 0.8616
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9363
  - pr_auc: 0.9450

### Dataset: wikipedia_politeness

- **Samples:** train 3024, val 432, test 864
- **Artifacts:** `d:\E_textbooks_And_Lecture_Notes\NLP\NLP10\Codes\QLoRa\outputs\full_run_002\run_002\wikipedia_politeness`

- **Validation Metrics:**
  - accuracy: 0.7500
  - f1: 0.7874
  - precision: 0.6849
  - recall: 0.9259
- **Test Metrics:**
  - accuracy: 0.6875
  - f1: 0.7384
  - precision: 0.6350
  - recall: 0.8819
- **Test Probabilistic Metrics:**
  - roc_auc: 0.8011
  - pr_auc: 0.8054

### Dataset: go_emotions

- **Samples:** train 6503, val 930, test 1859
- **Artifacts:** `d:\E_textbooks_And_Lecture_Notes\NLP\NLP10\Codes\QLoRa\outputs\full_run_002\run_002\go_emotions`

- **Validation Metrics:**
  - accuracy: 0.8849
  - f1: 0.8810
  - precision: 0.9124
  - recall: 0.8516
- **Test Metrics:**
  - accuracy: 0.8741
  - f1: 0.8714
  - precision: 0.8900
  - recall: 0.8536
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9353
  - pr_auc: 0.9457

### Dataset: civil_comments

- **Samples:** train 2303, val 330, test 659
- **Artifacts:** `d:\E_textbooks_And_Lecture_Notes\NLP\NLP10\Codes\QLoRa\outputs\full_run_002\run_002\civil_comments`

- **Validation Metrics:**
  - accuracy: 0.8788
  - f1: 0.8817
  - precision: 0.8613
  - recall: 0.9030
- **Test Metrics:**
  - accuracy: 0.8498
  - f1: 0.8588
  - precision: 0.8091
  - recall: 0.9149
- **Test Probabilistic Metrics:**
  - roc_auc: 0.9187
  - pr_auc: 0.9122

### Dataset: toxigen

- **Samples:** train 992, val 142, test 284
- **Artifacts:** `d:\E_textbooks_And_Lecture_Notes\NLP\NLP10\Codes\QLoRa\outputs\full_run_002\run_002\toxigen`

- **Validation Metrics:**
  - accuracy: 0.8732
  - f1: 0.8831
  - precision: 0.8193
  - recall: 0.9577
- **Test Metrics:**
  - accuracy: 0.7782
  - f1: 0.7893
  - precision: 0.7516
  - recall: 0.8310
- **Test Probabilistic Metrics:**
  - roc_auc: 0.8431
  - pr_auc: 0.8046

## Generated Files

- **Configuration:** `config.json`
- **Log File:** `experiment.log`
- **Per-dataset outputs:** one directory per dataset containing evaluation artifacts
- **Embeddings:** `*/embeddings/` (per dataset if saved)
- **Models:** `*/models/` (per dataset if saved)
