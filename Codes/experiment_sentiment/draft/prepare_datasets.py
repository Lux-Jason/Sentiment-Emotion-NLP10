#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and standardize multiple public datasets into Archive/datasets/{name}/{split}.csv

Supported datasets (key -> source):
- wikipedia_politeness -> convokit "wikipedia-politeness-corpus" (binary: polite vs impolite)
- go_emotions         -> HF "SetFit/go_emotions" (uses provided binary label if present)
- civil_comments      -> HF "google/civil_comments" (binary: toxicity >= 0.5)
- toxigen             -> HF "toxigen/toxigen-data" (config: annotations; binary: hate vs non-hate)

Output format for each split: CSV with header: text,label (label in {0,1})

Usage examples:
  python prepare_datasets.py --datasets wikipedia_politeness civil_comments --archive-root Archive --max-samples 5000
  python prepare_datasets.py --datasets all --archive-root Archive
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import sys
import csv
import random
from typing import List, Dict, Any, Tuple


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Tuple[str, int]]):
    _ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label'])
        for t, y in rows:
            # simple sanitization
            if t is None:
                continue
            t = str(t).replace('\r', ' ').replace('\n', ' ').strip()
            if t:
                w.writerow([t, int(y)])


def _train_dev_test_split(rows: List[Tuple[str, int]], seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    train = rows[:n_train]
    dev = rows[n_train:n_train+n_dev]
    test = rows[n_train+n_dev:]
    return train, dev, test


def dl_wikipedia_politeness(dst_dir: Path, max_samples: int | None = None) -> None:
    try:
        from convokit import Corpus, download
    except Exception as e:
        raise RuntimeError("convokit not installed. Please `pip install convokit`.") from e

    print("Downloading via convokit: wikipedia-politeness-corpus …")
    corpus_path = download("wikipedia-politeness-corpus")
    corpus = Corpus(filename=corpus_path)

    rows: List[Tuple[str, int]] = []
    # Utterance.meta often contains keys like 'Binary' or 'politeness_label'
    for utt in corpus.iter_utterances():
        text = getattr(utt, 'text', None)
        meta = getattr(utt, 'meta', {}) or {}
        y = None
        # try common patterns
        if isinstance(meta, dict):
            if 'Binary' in meta:
                # Some corpora store {'polite': bool, 'impolite': bool} or similar
                val = meta['Binary']
                if isinstance(val, dict):
                    if 'polite' in val:
                        y = 1 if bool(val['polite']) else 0
                    elif 'impolite' in val:
                        y = 0 if bool(val['impolite']) else 1
                elif isinstance(val, (str, int, bool)):
                    s = str(val).lower()
                    if s in ('polite', '1', 'true'):
                        y = 1
                    elif s in ('impolite', '0', 'false'):
                        y = 0
            if y is None and 'politeness_label' in meta:
                s = str(meta['politeness_label']).lower()
                y = 1 if 'polite' in s else 0 if 'impolite' in s else None
            if y is None and 'is_polite' in meta:
                y = 1 if bool(meta['is_polite']) else 0
        if text and y is not None:
            rows.append((text, y))
            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise RuntimeError("No labeled rows extracted from wikipedia-politeness-corpus. Label schema may differ.")

    train, dev, test = _train_dev_test_split(rows)
    _write_csv(dst_dir / 'train.csv', train)
    _write_csv(dst_dir / 'dev.csv', dev)
    _write_csv(dst_dir / 'test.csv', test)
    print(f"Saved wikipedia_politeness: train={len(train)} dev={len(dev)} test={len(test)}")


def dl_hf_go_emotions(dst_dir: Path, max_samples: int | None = None) -> None:
    from datasets import load_dataset
    print("Downloading HF: SetFit/go_emotions …")
    ds = load_dataset("SetFit/go_emotions")

    def std_split(split_name: str):
        if split_name not in ds:
            return []
        rows = []
        split = ds[split_name]
        # The SetFit/go_emotions dataset exposes one binary column per emotion + a text column.
        text_col = 'text' if 'text' in split.column_names else 'sentence' if 'sentence' in split.column_names else None
        pos_names = {
            'admiration','approval','gratitude','joy','love','optimism','pride','relief','amusement','excitement'
        }
        neg_names = {
            'anger','annoyance','disappointment','disapproval','disgust','embarrassment','fear','grief','nervousness','remorse','sadness'
        }
        if text_col is None:
            return rows
        # Reduce size early to avoid heavy random access; take a reasonable cap
        cap = max_samples * 3 if max_samples else len(split)
        cap = min(cap, len(split))
        small = split.select(range(cap)) if cap < len(split) else split
        # Convert to pandas for fast column ops
        import pandas as pd
        df = small.to_pandas()
        available_pos = [c for c in df.columns if c in pos_names]
        available_neg = [c for c in df.columns if c in neg_names]
        if not (available_pos or available_neg):
            return rows
        # Compute sums and filter unambiguous polarity
        pos_sum = df[available_pos].sum(axis=1) if available_pos else 0
        neg_sum = df[available_neg].sum(axis=1) if available_neg else 0
        pos_mask = (pos_sum > 0) & (neg_sum == 0)
        neg_mask = (neg_sum > 0) & (pos_sum == 0)
        pos_texts = df.loc[pos_mask, text_col].astype(str).tolist()
        neg_texts = df.loc[neg_mask, text_col].astype(str).tolist()
        for t in pos_texts:
            rows.append((t, 1))
            if max_samples and len(rows) >= max_samples:
                break
        if not max_samples or len(rows) < max_samples:
            remain = (max_samples - len(rows)) if max_samples else None
            for t in neg_texts[:remain]:
                rows.append((t, 0))
        return rows

    train = std_split('train')
    dev = std_split('validation') or std_split('dev')
    test = std_split('test')
    if not dev:
        # create split
        all_rows = train + test
        train, dev, test = _train_dev_test_split(all_rows)
    _write_csv(dst_dir / 'train.csv', train)
    _write_csv(dst_dir / 'dev.csv', dev)
    _write_csv(dst_dir / 'test.csv', test)
    print(f"Saved go_emotions: train={len(train)} dev={len(dev)} test={len(test)}")


def dl_hf_civil_comments(dst_dir: Path, max_samples: int | None = None) -> None:
    from datasets import load_dataset
    print("Downloading HF: google/civil_comments …")
    ds = load_dataset("google/civil_comments")

    def map_split(split_name: str):
        if split_name not in ds:
            return []
        split = ds[split_name]
        rows = []
        text_col = 'text' if 'text' in split.column_names else 'comment_text' if 'comment_text' in split.column_names else None
        for i in range(len(split)):
            t = split[text_col][i] if text_col else None
            y = None
            if 'toxicity' in split.column_names:
                try:
                    y = 1 if float(split['toxicity'][i]) >= 0.5 else 0
                except Exception:
                    y = None
            if y is None and 'toxic' in split.column_names:
                y = 1 if bool(split['toxic'][i]) else 0
            if t and y is not None:
                rows.append((t, y))
                if max_samples and len(rows) >= max_samples:
                    break
        return rows

    train = map_split('train')
    dev = map_split('validation') or map_split('dev')
    test = map_split('test')
    if not dev:
        all_rows = train + test
        train, dev, test = _train_dev_test_split(all_rows)
    _write_csv(dst_dir / 'train.csv', train)
    _write_csv(dst_dir / 'dev.csv', dev)
    _write_csv(dst_dir / 'test.csv', test)
    print(f"Saved civil_comments: train={len(train)} dev={len(dev)} test={len(test)}")


def _decode_maybe_bytes_literal(s: str) -> str:
    """Some toxigen fields look like "b'...text...'" string literals. Try to interpret and decode."""
    if not isinstance(s, str):
        return str(s)
    st = s.strip()
    if (st.startswith("b'") and st.endswith("'")) or (st.startswith('b"') and st.endswith('"')):
        import ast
        try:
            b = ast.literal_eval(st)
            if isinstance(b, (bytes, bytearray)):
                return b.decode('utf-8', errors='ignore')
        except Exception:
            pass
        # Fallback: strip leading b''
        return st[2:-1]
    return s


def dl_hf_toxigen(dst_dir: Path, max_samples: int | None = None) -> None:
    from datasets import load_dataset
    print("Downloading HF: toxigen/toxigen-data (annotations) …")
    ds = load_dataset("toxigen/toxigen-data", "annotations")

    # annotations config only has 'train' split. Map into train/dev/test here.
    split = ds['train']
    rows = []
    # text likely in 'Input.text'; label in 'Input.binary_prompt_label' (1 for hate)
    text_col = None
    for cand in ['Input.text', 'text', 'content']:
        if cand in split.column_names:
            text_col = cand
            break
    label_col = None
    for cand in ['Input.binary_prompt_label', 'is_hate', 'label']:
        if cand in split.column_names:
            label_col = cand
            break
    if text_col is None or label_col is None:
        raise RuntimeError("toxigen annotations: required text/label columns not found")
    # Reduce size early
    cap = max_samples * 3 if max_samples else len(split)
    cap = min(cap, len(split))
    small = split.select(range(cap)) if cap < len(split) else split
    import pandas as pd
    df = small.to_pandas()[[text_col, label_col]].copy()
    # Decode byte-literal strings
    df[text_col] = df[text_col].astype(str).map(_decode_maybe_bytes_literal)
    def map_label(v):
        if isinstance(v, bool):
            return 1 if v else 0
        try:
            iv = int(v)
            return 1 if iv == 1 else 0
        except Exception:
            s = str(v).lower()
            if s in ('hate','toxic','offensive','1','true'):
                return 1
            if s in ('non-hate','not_hate','non_toxic','0','false'):
                return 0
            return None
    df['label_bin'] = df[label_col].map(map_label)
    df = df.dropna(subset=['label_bin'])
    for _, row in df.iterrows():
        rows.append((row[text_col], int(row['label_bin'])))
        if max_samples and len(rows) >= max_samples:
            break
    train, dev, test = _train_dev_test_split(rows)
    _write_csv(dst_dir / 'train.csv', train)
    _write_csv(dst_dir / 'dev.csv', dev)
    _write_csv(dst_dir / 'test.csv', test)
    print(f"Saved toxigen: train={len(train)} dev={len(dev)} test={len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        help='Datasets to download: wikipedia_politeness go_emotions civil_comments toxigen or all')
    parser.add_argument('--archive-root', type=str, default='Archive', help='Archive root directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per split (for quick preparation)')
    args = parser.parse_args()

    # modest determinism
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    wanted = set([w.lower() for w in args.datasets])
    if 'all' in wanted:
        wanted = {'wikipedia_politeness','go_emotions','civil_comments','toxigen'}

    base = Path(args.archive_root) / 'datasets'
    _ensure_dir(base)

    if 'wikipedia_politeness' in wanted:
        dl_wikipedia_politeness(base / 'wikipedia_politeness', max_samples=args.max_samples)
    if 'go_emotions' in wanted:
        dl_hf_go_emotions(base / 'go_emotions', max_samples=args.max_samples)
    if 'civil_comments' in wanted:
        dl_hf_civil_comments(base / 'civil_comments', max_samples=args.max_samples)
    if 'toxigen' in wanted:
        dl_hf_toxigen(base / 'toxigen', max_samples=args.max_samples)

    print('All requested datasets prepared under', base)


if __name__ == '__main__':
    main()
