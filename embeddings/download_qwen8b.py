#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os, sys

repo_id = 'Qwen/Qwen3-Embedding-8B'
cache_dir = os.path.join(os.getcwd(), 'model_cache', 'Qwen3-Embedding-8B')
print('Downloading', repo_id, 'to', cache_dir)
try:
    path = snapshot_download(repo_id, cache_dir=cache_dir)
    print('Download finished. Path:', path)
    # list some files
    for root, dirs, files in os.walk(path):
        for f in files[:40]:
            print(os.path.join(root, f))
        break
except Exception as e:
    print('Download failed:', repr(e))
    sys.exit(2)

print('\nDone')
