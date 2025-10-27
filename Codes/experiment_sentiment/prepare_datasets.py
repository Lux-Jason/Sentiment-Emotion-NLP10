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
        # Prefer columns 'text' and 'label' if present
        text_col = 'text' if 'text' in split.column_names else 'sentence' if 'sentence' in split.column_names else None
        label_col = 'label' if 'label' in split.column_names else None
        if text_col and label_col:
            for i in range(len(split)):
                t = split[text_col][i]
                y = split[label_col][i]
                if isinstance(y, bool):
                    y = int(y)
                elif isinstance(y, int):
                    y = 1 if y == 1 else 0  # assume binary mapping
                else:
                    continue
                rows.append((t, y))
                if max_samples and len(rows) >= max_samples:
                    break
        else:
            # Fallback: try multi-label 'labels' with class names mapping if available
            labels_feat = split.features.get('labels') if hasattr(split, 'features') else None
            id2name = None
            try:
                if labels_feat and hasattr(labels_feat.feature, 'names'):
                    id2name = labels_feat.feature.names
            except Exception:
                pass
            pos_names = {
                'admiration','approval','gratitude','joy','love','optimism','pride','relief','amusement','excitement'
            }
            neg_names = {
                'anger','annoyance','disappointment','disapproval','disgust','embarrassment','fear','grief','nervousness','remorse','sadness'
            }
            if 'labels' in split.column_names and text_col:
                labels_list = split['labels']
                texts = split[text_col]
                for i, labs in enumerate(labels_list):
                    names = set()
                    if id2name:
                        names = {id2name[j] for j in labs}
                    # decide polarity only when unambiguous
                    is_pos = len(names & pos_names) > 0 and len(names & neg_names) == 0
                    is_neg = len(names & neg_names) > 0 and len(names & pos_names) == 0
                    if is_pos:
                        rows.append((texts[i], 1))
                    elif is_neg:
                        rows.append((texts[i], 0))
                    if max_samples and len(rows) >= max_samples:
                        break
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


def dl_hf_toxigen(dst_dir: Path, max_samples: int | None = None) -> None:
    from datasets import load_dataset
    print("Downloading HF: toxigen/toxigen-data (annotations) …")
    ds = load_dataset("toxigen/toxigen-data", "annotations")

    def map_split(split_name: str):
        if split_name not in ds:
            return []
        split = ds[split_name]
        rows = []
        text_col = 'text' if 'text' in split.column_names else 'content' if 'content' in split.column_names else None
        for i in range(len(split)):
            t = split[text_col][i] if text_col else None
            y = None
            # common fields: 'label' (0/1 or strings), 'is_hate' bool, 'toxicity' float
            if 'is_hate' in split.column_names:
                y = 1 if bool(split['is_hate'][i]) else 0
            if y is None and 'label' in split.column_names:
                v = split['label'][i]
                if isinstance(v, bool):
                    y = 1 if v else 0
                elif isinstance(v, int):
                    y = 1 if v == 1 else 0
                else:
                    s = str(v).lower()
                    if s in ('hate','toxic','offensive','1','true'):
                        y = 1
                    elif s in ('non-hate','not_hate','non_toxic','0','false'):
                        y = 0
            if y is None and 'toxicity' in split.column_names:
                try:
                    y = 1 if float(split['toxicity'][i]) >= 0.5 else 0
                except Exception:
                    y = None
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
