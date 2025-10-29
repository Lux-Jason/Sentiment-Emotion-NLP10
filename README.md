# Detecting "Pseudo-civility" Comments — Project Overview

 This repository collects implementations, experiments, and engineering practices for detecting "pseudo-civility": comments that look polite on the surface (polite words, hedging, softened phrasing) but communicate negative pragmatic intent (sarcasm, contempt, passive aggression, or veiled attacks).

 This README summarizes:

 - Characteristics of the four main datasets used in experiments;
 - Two primary modeling families and their training differences (classic ensemble/NN vs. QLoRA-driven embedding & fine-tuning);
 - Recommended comparison and ablation experiments;
 - The engineering advantages of QLoRA in this project;
 - Current data limitations and next steps (note: we currently cannot complete full end-to-end pseudo-civility training due to dataset constraints).

 The content below is adapted from `Codes/final_implementation_summary.md` and `Codes/QLoRa/README_MODULAR.md`. It intentionally omits runtime instructions and tests.

 ## 1. Four datasets — quick summary

 The project commonly uses four public datasets (stored under `Codes/Archive/datasets` in this repo). Each dataset provides a different supervision signal used to construct features for pseudo-civility detection:

 - civil_comments — toxic / abusive labels for online comments. Useful as an "uncivil" anchor and for noise-robust training.
 - go_emotions — multi-label emotion annotations. Useful for building emotion features (S) and detecting emotion-polarity conflicts that can indicate sarcasm.
 - toxigen — collections and annotations of generated or implicit toxic content. Helps increase sensitivity to indirect or subtle attacks.
 - wikipedia_politeness — politeness / mitigation patterns (e.g., Stanford Politeness style data). Used to estimate PolitenessScore (P) and detect polite surface forms.

 Combined, these sources let us extract three core signals for each example: Politeness (P), emotion/toxicity (S), and a semantic–emotion inconsistency measure (I) that often marks pseudo-civility.

 ## 2. Two modeling families and training differences

 We implemented two complementary modeling routes.

 1) Classic ensemble & lightweight neural models (engineering-first)

 - Composition: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, RBF-SVM fused into an ensemble, plus a compact MLP (e.g., 256-128-64).
 - Inputs: sentence embeddings or TF–IDF combined with handcrafted features (P, S, I, sarcasm cues, meta features).
 - Pros: low resource requirements, fast training, strong interpretability (coefficients, SHAP). Good for small-to-medium data and fast iteration.

 2) QLoRA-driven embeddings and LoRA fine-tuning (performance-first)

 - Composition: an embedding-based path (use embeddings + classic classifier) and a QLoRA path (load a quantized base model and fine-tune LoRA adapters).
 - Key idea: load the base model in low-bit precision (e.g., 4-bit) and only train low-rank LoRA adapters; this saves memory while retaining high performance.
 - Pros: much lower GPU memory and storage costs than full-precision fine-tuning; adapters are small, versionable, and reusable. Embedding caching enables fast iteration; LoRA micro-tuning raises the performance ceiling when resources allow.

 Recommendation: start with embedding + classic classifiers for quick validation; use QLoRA adapter fine-tuning when seeking the highest performance and when modest GPU resources are available (e.g., 8–12 GB cards).

 ## 3. Comparison and ablation suggestions

 To validate feature and modeling choices, run these comparisons:

 - Baseline: sentence embedding + linear head.
 - +Politeness: add P features.
 - +Emotion: add S features.
 - +Inconsistency: add I = |S − SP| (or a signed conflict indicator).
 - +Sarcasm/Meta: add sarcasm cues, punctuation/emoji/length features.
 - Model-level: LR / XGBoost / RF / MLP / Transformer (single-task vs. multi-task with politeness/emotion/toxicity auxiliary heads).
 - Strategy-level: two-stage (candidate filtering → subset binary classifier) vs. end-to-end three-class classification.

 These ablations quantify how much each signal and modeling decision helps the pseudo-civility class (primary metric: Pseudo-Civil F1).

 ## 4. Why QLoRA matters — core engineering advantages

 QLoRA is central to the high-performance, low-cost workflow in this project for several reasons:

 - Low memory footprint: 4-bit quantized model loading plus training of tiny LoRA adapters drastically reduces GPU memory needs.
 - Cost and speed: training is faster and less memory-bound than full-precision fine-tuning, especially with gradient accumulation and AMP.
 - Parameter efficiency: adapters are small and can be saved/loaded independently, enabling modular versioning and reuse.
 - Flexibility: you can precompute and cache embeddings for rapid experiments, and still switch to LoRA micro-tuning to push performance further.
 - Robust engineering: the implementation supports graceful fallbacks (4-bit → 8-bit → FP32), and standard stabilization techniques (small LR, warmup, early stop, scheduler).

 In short, QLoRA delivers a practical tradeoff between large-model capability and constrained compute resources, making it a suitable bridge from research POC to production-friendly workflows.

 ## 5. Current limitations and next steps (important)

 The primary bottleneck for delivering a complete, end-to-end pseudo-civility classifier is the lack of a large, high-quality annotated dataset specifically labeled for pseudo-civility (i.e., polite surface form + negative pragmatic intent). While we use multi-source signals (politeness, emotion, toxicity, sarcasm datasets, and platform removal weak labels) to construct training sets, noise and label mismatch limit end-to-end performance and generalization.

 Planned next steps:

 1. Build a dedicated annotated pseudo-civility dataset (multi-round review, clear guidelines, example catalogs).
 2. Explore weak- and semi-supervised learning: combine platform weak labels (removed flags) with multi-source anchors (P/S/Toxicity) for self-training, contrastive learning, or noise-robust losses.
 3. Improve sarcasm/irony detectors and discourse-act models to reduce false positives on reasoned rebuttals.
 4. Extend to multilingual and cross-lingual transfer settings.
 5. Evaluate adversarial robustness, calibration (ECE), and domain adaptation for safe deployment.

 The codebase already stores artifacts such as embeddings, adapters, and reports to enable rapid iteration once stronger labeled data become available.

 ## 6. Closing note

This repository documents the design choices, dataset dependencies, and engineering rationale (especially around QLoRA) for pseudo-civility research. It also explicitly highlights that, given current dataset limitations, a final end-to-end pseudo-civility training pipeline is not yet completed. Future work will focus on data collection and weak-supervision methods combined with QLoRA micro-tuning to reach production-quality performance.