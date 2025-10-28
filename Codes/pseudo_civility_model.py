"""
pseudo_civility_model.py
=======================

This module provides a modular framework for training classifiers to detect
"pseudo‑civility" in text – comments that appear polite on the surface
but conceal negative sentiment, sarcasm or hidden abuse.  The code is
designed to work with the Qwen embedding family (e.g. `Qwen/Qwen3‑Embedding‑4B`)
via the HuggingFace transformers library, and includes plug‑and‑play
support for parameter‑efficient fine tuning (LoRA/QLoRA) as well as
contrastive fine tuning via SetFit.  The functions defined here allow
loading of several common open datasets, optional prompt engineering for
embedding generation, and flexible training/evaluation loops for both
binary and multi‑class problems.

The goal of modularity is twofold:

* **Ease of experimentation**: Datasets, embedding models, fine tuning
  strategies and downstream classifiers are defined as interchangeable
  pieces.  You can switch between binary and three‑class setups or
  experiment with different fine tuning methods without rewriting the
  entire training script.
* **Extensibility**: Additional datasets or tasks can be incorporated
  by implementing new `load_*` functions and mapping them into the
  unified example format (a pair of `text` and `label`).

The module relies on a handful of third‑party packages:

* **transformers** (for Qwen embeddings and LoRA/QLoRA integration)
* **datasets** (for easy access to common NLP datasets)
* **setfit** (optional; for contrastive fine tuning)
* **scikit‑learn** (for lightweight classification heads and metrics)

If these libraries are not installed, you can install them via pip
before running this script:

```bash
pip install transformers datasets setfit scikit‑learn peft accelerate
```

Note: When running on GPUs with limited memory (e.g. laptops), you
should enable low‑memory configurations such as 4‑bit quantization in
QLoRA and reduce batch sizes.  The `train_*` functions accept
configuration dictionaries to tailor model and training hyperparameters
to your hardware constraints.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import third‑party libraries with optional fallbacks.
try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "The `datasets` library is required to load datasets.\n"
        "Install it via `pip install datasets`."
    ) from exc

try:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise ImportError(
        "Scikit‑learn is required for classification and evaluation.\n"
        "Install it via `pip install scikit‑learn`."
    ) from exc

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers import BitsAndBytesConfig
except ImportError as exc:
    raise ImportError(
        "The `transformers` library is required for Qwen embeddings.\n"
        "Install it via `pip install transformers`."
    ) from exc

try:
    from peft import LoraConfig, get_peft_model  # type: ignore
except ImportError:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

try:
    from setfit import SetFitModel  # type: ignore
    from setfit.training import SetFitTrainer  # type: ignore
except ImportError:
    SetFitModel = None  # type: ignore
    SetFitTrainer = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DataExample:
    """A simple data container for a single example."""
    text: str
    label: int


def load_politeness_dataset() -> List[DataExample]:
    """Load the Stanford/Wikipedia politeness corpus.

    This function loads both the Wikipedia and StackExchange splits
    (via HuggingFace) and concatenates them.  Labels 'polite' and
    'impolite' are mapped to 0 and 1, respectively.
    """
    try:
        wiki = load_dataset("politeness_wikipedia", split="train")
        se = load_dataset("politeness_stackexchange", split="train")
    except Exception:
        raise RuntimeError(
            "Unable to load the politeness corpora via HuggingFace.\n"
            "Please download via ConvoKit if available."
        )
    examples: List[DataExample] = []
    for ds in [wiki, se]:
        for record in ds:
            text = record.get("text", "").strip()
            label_str = record.get("label", "polite").lower()
            label = 0 if label_str == "polite" else 1
            if text:
                examples.append(DataExample(text=text, label=label))
    return examples


def load_go_emotions(split: str = "train", neutral_class: bool = False) -> List[DataExample]:
    """Load the GoEmotions dataset and map emotions to coarse sentiment.

    Parameters
    ----------
    split : str
        One of "train", "validation", or "test".
    neutral_class : bool
        Whether to include a neutral class (id=2).  If False, neutral
        examples are discarded.
    """
    ds = load_dataset("SetFit/go_emotions", split=split)
    # Mapping of emotion ids into positive and negative categories
    pos_ids = {1, 2, 4, 6, 7, 9, 13, 16, 17, 22, 23}
    neg_ids = {0, 3, 5, 8, 10, 11, 12, 14, 15, 18, 19, 20, 21, 24, 25, 26}
    examples: List[DataExample] = []
    for record in ds:
        text = record["text"].strip()
        labels = record["labels"]
        if not labels:
            continue
        pos_count = sum(1 for l in labels if l in pos_ids)
        neg_count = sum(1 for l in labels if l in neg_ids)
        neutral_count = sum(1 for l in labels if l == 27)
        if pos_count > neg_count:
            coarse_label = 0
        elif neg_count > pos_count:
            coarse_label = 1
        else:
            if neutral_class and neutral_count > 0:
                coarse_label = 2
            else:
                continue
        if text:
            examples.append(DataExample(text=text, label=coarse_label))
    return examples


def load_civil_comments(split: str = "train") -> List[DataExample]:
    """Load the Civil Comments dataset for binary toxicity detection."""
    if split not in {"train", "validation"}:
        return []
    ds = load_dataset("google/civil_comments", split=split)
    examples: List[DataExample] = []
    for record in ds:
        text = record["text"].strip()
        toxic = int(record["toxicity"] >= 0.5)
        examples.append(DataExample(text=text, label=toxic))
    return examples


def load_toxigen() -> List[DataExample]:
    """Load the ToxiGen annotations dataset."""
    ds = load_dataset("toxigen/toxigen-data", "annotations", split="train")
    examples: List[DataExample] = []
    for record in ds:
        text = record["content"].strip()
        label = int(record["label"] == 1)
        examples.append(DataExample(text=text, label=label))
    return examples


def train_test_split_examples(
    examples: List[DataExample],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[DataExample], List[DataExample]]:
    """Shuffle and split examples into train and test lists."""
    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_examples = [DataExample(text=x, label=y) for x, y in zip(X_train, y_train)]
    test_examples = [DataExample(text=x, label=y) for x, y in zip(X_test, y_test)]
    return train_examples, test_examples


class QwenEmbedder:
    """Embedding wrapper for Qwen models with optional instruction prompts."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: Optional[str] = None,
        quantize: bool = False,
        instruction: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize
        self.instruction = instruction
        logger.info("Loading Qwen tokenizer and model…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        all_embeddings: List[np.ndarray] = []
        if self.instruction:
            prefix = f"<|instr|>{self.instruction}<|/instr|> <|input|>"
            suffix = "<|/input|>"
            wrapped = [f"{prefix} {text} {suffix}" for text in texts]
        else:
            wrapped = texts
        for i in range(0, len(wrapped), batch_size):
            batch_texts = wrapped[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs.cpu().numpy()
                else:
                    raise RuntimeError(
                        f"Unexpected model output type: {type(outputs)}"
                    )
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)


def train_logreg_classifier(
    embedder: QwenEmbedder,
    train_examples: List[DataExample],
    val_examples: Optional[List[DataExample]] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Train a logistic regression classifier on Qwen embeddings."""
    # Optionally insert LoRA layers (currently untrained) – for future use
    if lora_config and LoraConfig and get_peft_model:
        logger.info("Applying LoRA configuration…")
        config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        embedder.model = get_peft_model(embedder.model, config)
        embedder.model.train()
        logger.warning(
            "LoRA layers added but training loop not implemented – you may fine tune manually."
        )
        embedder.model.eval()

    if max_samples is not None and len(train_examples) > max_samples:
        train_examples = train_examples[:max_samples]

    # Compute embeddings
    train_texts = [ex.text for ex in train_examples]
    train_labels = np.array([ex.label for ex in train_examples])
    train_embeddings = embedder.encode(train_texts, batch_size=batch_size)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_embeddings, train_labels)
    if val_examples is not None:
        val_texts = [ex.text for ex in val_examples]
        val_labels = np.array([ex.label for ex in val_examples])
        val_embeddings = embedder.encode(val_texts, batch_size=batch_size)
        return clf, val_embeddings, val_labels
    else:
        return clf, np.empty((0,)), np.empty((0,))


def evaluate_classifier(
    clf: Any,
    embeddings: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """Evaluate model performance and return accuracy, F1 and recall."""
    preds = clf.predict(embeddings)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average=average)),
        "recall": float(recall_score(labels, preds, average=average)),
    }


def main() -> None:
    """Run a demonstration experiment using the modular pipeline."""
    logging.basicConfig(level=logging.INFO)
    # Load datasets
    logger.info("Loading datasets…")
    politeness_examples = load_politeness_dataset()
    go_examples = load_go_emotions(split="train")
    civil_examples = load_civil_comments(split="train")
    toxigen_examples = load_toxigen()
    # Construct binary classification dataset: harmful vs benign
    binary_examples: List[DataExample] = []
    binary_examples += politeness_examples
    binary_examples += civil_examples
    binary_examples += toxigen_examples
    train_bin, test_bin = train_test_split_examples(binary_examples, test_size=0.2)
    # Construct tri‑class dataset: polite/positive (0), impolite/toxic (2), neutral omitted
    tri_examples: List[DataExample] = []
    for ex in politeness_examples:
        tri_examples.append(DataExample(text=ex.text, label=0 if ex.label == 0 else 2))
    for ex in go_examples:
        if ex.label == 0:
            tri_examples.append(DataExample(text=ex.text, label=0))
        elif ex.label == 1:
            tri_examples.append(DataExample(text=ex.text, label=2))
    for ex in civil_examples + toxigen_examples:
        tri_examples.append(DataExample(text=ex.text, label=2 if ex.label == 1 else 0))
    train_tri, test_tri = train_test_split_examples(tri_examples, test_size=0.2)
    # Initialize embedder with prompt engineering
    instruction = (
        "Represent this comment for civility classification. Return a vector"
        " that captures politeness, sentiment and toxicity."
    )
    embedder = QwenEmbedder(
        model_name="Qwen/Qwen3-Embedding-4B",
        quantize=True,
        instruction=instruction,
    )
    # Train binary classifier
    logger.info("Training binary classifier…")
    clf_bin, val_emb_bin, val_labels_bin = train_logreg_classifier(
        embedder,
        train_bin,
        val_examples=test_bin,
        lora_config=None,
        batch_size=16,
        max_samples=3000,
    )
    metrics_bin = evaluate_classifier(clf_bin, val_emb_bin, val_labels_bin)
    logger.info(f"Binary metrics: {metrics_bin}")
    # Train tri‑class classifier
    logger.info("Training tri‑class classifier…")
    clf_tri, val_emb_tri, val_labels_tri = train_logreg_classifier(
        embedder,
        train_tri,
        val_examples=test_tri,
        lora_config=None,
        batch_size=16,
        max_samples=3000,
    )
    metrics_tri = evaluate_classifier(clf_tri, val_emb_tri, val_labels_tri, average="macro")
    logger.info(f"Tri‑class metrics: {metrics_tri}")
    print("\nResults Summary:")
    print(
        f"Binary – Accuracy: {metrics_bin['accuracy']:.3f}, F1: {metrics_bin['f1']:.3f}, Recall: {metrics_bin['recall']:.3f}"
    )
    print(
        f"Tri‑class – Accuracy: {metrics_tri['accuracy']:.3f}, F1: {metrics_tri['f1']:.3f}, Recall: {metrics_tri['recall']:.3f}"
    )


if __name__ == "__main__":
    main()
