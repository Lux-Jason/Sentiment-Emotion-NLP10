# Qwen模型下载指南

本指南帮助您下载Qwen的4B和8B嵌入模型，以便后续根据设备情况选择使用。

## 快速开始

### 下载所有模型（推荐）
```bash
cd embeddings
python download_helper.py
```
这将同时下载4B和8B两个模型，您可以根据设备情况选择使用。

### 选择性下载

#### 只下载4B模型
```bash
python download_helper.py --model 4B
```

#### 只下载8B模型
```bash
python download_helper.py --model 8B
```

### 查看模型对比
```bash
python download_helper.py --compare
```

## 模型选择建议

| 设备配置 | 推荐模型 | 原因 |
|---------|---------|------|
| GPU内存 < 12GB | 4B | 内存占用小，速度快 |
| GPU内存 >= 16GB | 8B | 性能最佳，MTEB排行榜第一 |
| CPU训练 | 4B | 资源占用更少 |

## 模型详细信息

### Qwen3-Embedding-4B
- **模型大小**: ~8GB
- **内存需求**: 8-10GB
- **C-MTEB中文评分**: 72.27
- **MTEB多语言评分**: 69.45
- **特点**: 速度快，资源占用少，适合大多数用户

### Qwen3-Embedding-8B
- **模型大小**: ~16GB
- **内存需求**: 16GB+
- **C-MTEB中文评分**: 73.84
- **MTEB多语言评分**: 70.58
- **特点**: 性能最佳，MTEB多语言排行榜第一

## 使用方法

下载完成后，您可以通过以下方式使用模型：

### 方法1: 使用config.py
```python
from config import load_model

# 使用4B模型
import config
config.MODEL_SIZE = "4B"
model = config.load_model()

# 使用8B模型
config.MODEL_SIZE = "8B"
model = config.load_model()
```

### 方法2: 直接加载
```python
from sentence_transformers import SentenceTransformer

# 加载4B模型
model_4b = SentenceTransformer(
    './model_cache/Qwen3-Embedding-4B',
    trust_remote_code=True
)

# 加载8B模型
model_8b = SentenceTransformer(
    './model_cache/Qwen3-Embedding-8B', 
    trust_remote_code=True
)
```

## 故障排除

### 下载失败
1. 检查网络连接
2. 尝试使用魔法上网
3. 使用 `--no-mirror` 参数从官方源下载

### 内存不足
- 如果GPU内存不足，可以设置 `device='cpu'` 使用CPU
- 考虑使用4B模型替代8B模型

### 依赖问题
确保安装了所需依赖：
```bash
pip install -r requirements.txt
```

## 缓存位置
模型默认下载到 `./model_cache/` 目录下：
- 4B模型: `./model_cache/Qwen3-Embedding-4B/`
- 8B模型: `./model_cache/Qwen3-Embedding-8B/`

## 其他下载选项

### 使用官方源（不使用镜像）
```bash
python download_helper.py --no-mirror
```

### 查看帮助
```bash
python download_helper.py --help
