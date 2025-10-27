# Experiment: Sentiment classification using Qwen3-Embedding (prefer 4B)

This folder contains a small experiment pipeline that:

- Loads the ChnSentiCorp dataset from the repository Archive folder.
- Uses the locally downloaded Qwen3-Embedding-4B snapshot first (if present) to compute embeddings; if not found, falls back to 8B.
- Trains a scikit-learn LogisticRegression classifier on the embeddings.
- Evaluates and saves the trained classifier and optional embedding cache.

Quick start (using conda env `chemist`):

1. Activate your environment or use conda run:
   conda activate chemist

2. Install any missing Python packages (if needed):
   pip install -r embeddings/requirements.txt

3. Run training (quick test with a subset):
   python train_sentiment.py --data-dir "Archive/chinesenlpdataset/ChnSentiCorp/ChnSentiCorp" --max-samples 2000 --batch-size 16 --device cuda --offline

Replace --max-samples with a larger number or omit it to use the full dataset.

Artifacts produced:

- model.joblib         : saved scikit-learn classifier
- embeddings_cache.npz : optional saved embeddings (train/dev/test)
- results.txt         : evaluation summary

Notes:

- The script will try to find a local Qwen3-Embedding-4B snapshot under `model_cache` first; if missing, it tries 8B. Both avoid re-downloading.
- If no local snapshot is found, it will fall back to using the `embeddings.config.load_model()` helper with a 4B preference, which may attempt to download using ModelScope/HuggingFace.
- On Windows/CUDA, if embedding seems stuck, try: smaller `--batch-size` (e.g., 8), `--offline`, or set env vars `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1`.

If you want me to run the experiment now in your `chemist` environment with a quick subset for verification, tell me and I'll run it.
