#!/usr/bin/env python3
"""Train a sentiment classifier using Qwen3-Embedding embeddings (prefer 4B).

This script:
- loads ChnSentiCorp TSV files (train/dev/test) from Archive
- finds a local Qwen3-Embedding-4B snapshot under model_cache first (preferred)
- if 4B not found, falls back to Qwen3-Embedding-8B local snapshot
- computes embeddings with SentenceTransformer
- trains a LogisticRegression classifier and evaluates it
- saves model and optional embeddings cache

Run example:
  python train_sentiment.py --data-dir "Archive/chinesenlpdataset/ChnSentiCorp/ChnSentiCorp" --max-samples 2000
"""
from pathlib import Path
import argparse
import os
import sys
import numpy as np
import joblib
import torch
from tqdm import tqdm
import warnings
import logging

# Reduce TensorFlow verbosity and suppress deprecation warnings coming from TF internals
# Safer defaults to avoid hangs on Windows/HF tokenizers and reduce thread contention
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # errors and warnings only
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass
except Exception:
    # TensorFlow may not be installed in some envs; ignore if import fails
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def read_tsv(path, max_samples=None):
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            label = parts[0].strip()
            text = parts[1].strip()
            if label == '1':
                labels.append(1)
            else:
                labels.append(0)
            texts.append(text)
    return texts, np.array(labels, dtype=int)


def find_local_qwen_snapshot(root: Path, size_preference=("4B", "8B")):
    """Search for a local Qwen3-Embedding-{size} snapshot under root/model_cache.

    Tries sizes in order of size_preference, returns the first found snapshot path or None.
    """
    if not root.exists():
        return None

    model_markers = {
        "4B": 'models--Qwen--Qwen3-Embedding-4B',
        "8B": 'models--Qwen--Qwen3-Embedding-8B',
    }

    for sz in size_preference:
        marker = model_markers.get(sz)
        if not marker:
            continue
        for p in root.rglob(marker):
            # .../models--Qwen--Qwen3-Embedding-4B|8B/snapshots/<hash>
            try:
                snapshots_dir = p / 'snapshots'
                if snapshots_dir.exists():
                    for snap in snapshots_dir.iterdir():
                        if snap.is_dir():
                            return str(snap)
            except Exception:
                continue
    return None


def batch_encode(model, texts, batch_size=16, device='cpu'):
    # Safer manual batching to avoid multiprocessing hangs on Windows
    # Iterate through texts in chunks and call model.encode on each chunk
    if not texts:
        return np.empty((0, 0))

    embeddings_list = []
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    for idx in tqdm(range(n_batches), desc="Encoding batches", unit="batch"):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        # disable internal progress bar for each small batch
        batch_emb = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings_list.append(batch_emb)

    try:
        embeddings = np.vstack(embeddings_list)
    except Exception:
        # fallback: concatenate
        embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='Archive/chinesenlpdataset/ChnSentiCorp/ChnSentiCorp',
                        help='Path to ChnSentiCorp folder containing train.tsv/dev.tsv/test.tsv')
    parser.add_argument('--model-cache-root', type=str, default='model_cache',
                        help='Root folder where models were downloaded (default: model_cache)')
    parser.add_argument('--device', type=str, default=None, help="Device to run embeddings on (cpu or cuda). If omitted, auto-detect GPU.")
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per split (for quick tests). Omit to use full dataset.')
    parser.add_argument('--out-dir', type=str, default='Codes/experiment_sentiment', help='Output folder')
    parser.add_argument('--batch-size', type=int, default=16, help='Encoding batch size (lower on GPU to avoid stalls)')
    parser.add_argument('--offline', action='store_true', help='Force HF/Transformers offline mode to avoid network calls')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / 'train.tsv'
    dev_path = data_dir / 'dev.tsv'
    test_path = data_dir / 'test.tsv'

    if not train_path.exists():
        print(f"Train file not found: {train_path}")
        sys.exit(1)

    print("Loading data...")
    X_train, y_train = read_tsv(train_path, max_samples=args.max_samples)
    if dev_path.exists():
        X_dev, y_dev = read_tsv(dev_path, max_samples=args.max_samples)
    else:
        X_dev, y_dev = [], np.array([], dtype=int)
    if test_path.exists():
        X_test, y_test = read_tsv(test_path, max_samples=args.max_samples)
    else:
        X_test, y_test = [], np.array([], dtype=int)

    print(f"Train samples: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

    # Decide device: prefer user arg, else auto-detect CUDA
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    # Optional offline mode to avoid any network IO that might cause stalls
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print("Offline mode enabled for HF hub and transformers.")

    # Try to find a local snapshot for Qwen3-Embedding under model_cache
    # Preference: 4B -> 8B
    local_snapshot = find_local_qwen_snapshot(Path(args.model_cache_root), size_preference=("4B", "8B"))
    model = None
    try:
        if local_snapshot:
            print(f"Found local Qwen snapshot: {local_snapshot}. Loading from local snapshot to avoid re-download.")
            from sentence_transformers import SentenceTransformer
            try:
                model = SentenceTransformer(local_snapshot, device=device, trust_remote_code=True)
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'cuda out of memory' in msg:
                    print("GPU out of memory when loading model. Falling back to CPU to continue (this will be slower).")
                    model = SentenceTransformer(local_snapshot, device='cpu', trust_remote_code=True)
                else:
                    raise
        else:
            print("Local snapshot not found. Falling back to embeddings.config.load_model() with 4B preference.")
            # Import the project's helper to load the model (this may attempt to download)
            # add repo root (…/NLP10) to sys.path
            try:
                repo_root = Path(__file__).parents[2]
                if str(repo_root) not in sys.path:
                    sys.path.append(str(repo_root))
            except Exception:
                pass
            import embeddings.config as econf
            # Force preference to 4B when using helper
            try:
                econf.MODEL_SIZE = "4B"
            except Exception:
                pass
            try:
                model = econf.load_model(device=device)
            except Exception as e:
                msg = str(e).lower()
                if 'out of memory' in msg or 'cuda out of memory' in msg:
                    print("GPU out of memory when loading model. Falling back to CPU to continue (this will be slower).")
                    model = econf.load_model(device='cpu')
                else:
                    raise
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        sys.exit(1)

    # Compute embeddings
    print("Computing embeddings for train set...")
    emb_train = batch_encode(model, X_train, batch_size=args.batch_size, device=args.device)

    emb_dev = None
    emb_test = None
    if len(X_dev) > 0:
        print("Computing embeddings for dev set...")
        emb_dev = batch_encode(model, X_dev, batch_size=args.batch_size, device=args.device)
    if len(X_test) > 0:
        print("Computing embeddings for test set...")
        emb_test = batch_encode(model, X_test, batch_size=args.batch_size, device=args.device)

    # Ensure dev/test embeddings are arrays (savez_compressed can't accept None)
    if emb_dev is None:
        emb_dev = np.empty((0, emb_train.shape[1]))
        y_dev = np.array([], dtype=int)
    if emb_test is None:
        emb_test = np.empty((0, emb_train.shape[1]))
        y_test = np.array([], dtype=int)

    # Save embedding cache
    cache_path = out_dir / 'embeddings_cache.npz'
    print(f"Saving embeddings to {cache_path}")
    np.savez_compressed(cache_path, train=emb_train, y_train=y_train,
                        dev=emb_dev, y_dev=y_dev, test=emb_test, y_test=y_test)

    # Train a simple classifier
    print("Training classifier (LogisticRegression)...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(emb_train, y_train)

    # Evaluate
    results = {}
    if emb_dev is not None:
        y_pred_dev = clf.predict(emb_dev)
        acc_dev = accuracy_score(y_dev, y_pred_dev)
        results['dev'] = {
            'accuracy': acc_dev,
            'report': classification_report(y_dev, y_pred_dev, digits=4)
        }
        print(f"Dev accuracy: {acc_dev:.4f}")
        print(results['dev']['report'])
    if emb_test is not None:
        y_pred_test = clf.predict(emb_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        results['test'] = {
            'accuracy': acc_test,
            'report': classification_report(y_test, y_pred_test, digits=4)
        }
        print(f"Test accuracy: {acc_test:.4f}")
        print(results['test']['report'])

    # Save classifier
    model_out = out_dir / 'model.joblib'
    print(f"Saving classifier to {model_out}")
    joblib.dump({'clf': clf, 'results': results}, model_out)

    # Save a brief results file
    results_path = out_dir / 'results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Train samples: {len(X_train)}\n")
        if 'dev' in results:
            f.write(f"Dev accuracy: {results['dev']['accuracy']:.4f}\n")
            f.write(results['dev']['report'] + '\n')
        if 'test' in results:
            f.write(f"Test accuracy: {results['test']['accuracy']:.4f}\n")
            f.write(results['test']['report'] + '\n')

    print("Done. Artifacts saved under:", out_dir)


if __name__ == '__main__':
    main()
