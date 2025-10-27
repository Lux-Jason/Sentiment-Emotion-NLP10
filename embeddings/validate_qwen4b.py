#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import numpy as np

# find local snapshot
root = Path.cwd()
model_marker = 'models--Qwen--Qwen3-Embedding-4B'
found = None
for p in root.rglob(model_marker):
    snaps = p / 'snapshots'
    if snaps.exists():
        for s in snaps.iterdir():
            if s.is_dir():
                found = str(s)
                break
    if found:
        break

if not found:
    print('No local snapshot found under model_cache. Please run download_qwen4b.py first.')
    sys.exit(2)

print('Found snapshot:', found)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print('Please install sentence-transformers in the active environment:', e)
    sys.exit(2)

# choose device
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

print('Loading model... (this may take a moment)')
model = SentenceTransformer(found, device=device, trust_remote_code=True)
print('Model loaded.')

texts = [
    '这是一个非常好的产品，我很喜欢。',
    '服务非常差，再也不会来了。',
    '还可以，有待改进。'
]
embs = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
print('Embeddings shape:', embs.shape)
for i, e in enumerate(embs):
    norm = np.linalg.norm(e)
    print(f'Text {i+1} norm: {norm:.4f}')

print('Done')
