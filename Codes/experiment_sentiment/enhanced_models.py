"""
Enhanced models module implementing advanced training methods from Qwen-4B documentation.
Includes SetFit contrastive learning, LoRA/QLoRA, and multi-task learning.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from dataclasses import dataclass

# Import with fallbacks
try:
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.util import pytorch_cos_sim
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers sentence-transformers")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not available. Install with: pip install peft")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_recall_curve, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)

# Stability settings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass
class ContrastiveExample:
    """Contrastive learning example pair."""
    anchor: str
    positive: str
    negative: str
    label: int


class EnhancedQwenEmbedder:
    """Enhanced Qwen embedding model with SetFit and advanced features."""
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.model.model_name
        self.device = self._get_device()
        self.quantize = config.model.quantize
        self.batch_size = config.model.batch_size
        self.use_instruction_prompt = config.model.use_instruction_prompt
        self.instruction = config.model.instruction
        
        # Advanced features
        self.use_lora = config.model.use_lora
        self.use_contrastive = config.model.use_contrastive
        self.use_setfit = config.model.use_setfit
        
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.model.device:
            return self.config.model.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the embedding model with advanced features."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        logger.info(f"Loading enhanced embedding model: {self.model_name}")
        
        # Try to find local snapshot first
        local_snapshot = self._find_local_snapshot()
        
        if local_snapshot:
            logger.info(f"Using local snapshot: {local_snapshot}")
            try:
                self.sentence_model = SentenceTransformer(
                    local_snapshot, 
                    device=self.device, 
                    trust_remote_code=True
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load local snapshot: {e}")
        
        # Fallback to transformers with advanced features
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            if self.quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                # Prepare for PEFT training
                if self.use_lora and PEFT_AVAILABLE:
                    self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
            
            self.model.eval()
            
            # Apply LoRA if configured
            if self.use_lora and PEFT_AVAILABLE:
                self._apply_lora()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _find_local_snapshot(self) -> Optional[str]:
        """Find local model snapshot in cache."""
        cache_root = Path(self.config.model.model_cache_root)
        if not cache_root.exists():
            return None
        
        model_markers = {
            "4B": 'models--Qwen--Qwen3-Embedding-4B',
            "8B": 'models--Qwen--Qwen3-Embedding-8B',
        }
        
        for size, marker in model_markers.items():
            if size in self.model_name:
                for p in cache_root.rglob(marker):
                    snapshots_dir = p / 'snapshots'
                    if snapshots_dir.exists():
                        for snap in snapshots_dir.iterdir():
                            if snap.is_dir():
                                return str(snap)
        return None
    
    def _apply_lora(self):
        """Apply enhanced LoRA configuration."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, skipping LoRA")
            return
        
        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="FEATURE_EXTRACTION"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Enhanced LoRA configuration applied (r={self.config.model.lora_r}, alpha={self.config.model.lora_alpha})")
    
    def _prepare_text_with_instruction(self, texts: List[str]) -> List[str]:
        """Prepare texts with enhanced instruction prompts."""
        if not self.use_instruction_prompt:
            return texts
        
        prefix = f"<|instr|>{self.instruction}<|/instr|> <|input|>"
        suffix = "<|/input|>"
        return [f"{prefix} {text} {suffix}" for text in texts]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.empty((0, 0))
        
        # Use SentenceTransformer if available
        if self.sentence_model:
            return self._encode_with_sentence_transformer(texts)
        
        # Use native transformers
        return self._encode_with_transformers(texts)
    
    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Alias for encode method to maintain compatibility."""
        return self.encode(texts)
    
    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode using SentenceTransformer with enhanced features."""
        prepared_texts = self._prepare_text_with_instruction(texts)
        
        embeddings = self.sentence_model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode using native transformers with enhanced pooling."""
        prepared_texts = self._prepare_text_with_instruction(texts)
        all_embeddings = []
        
        for i in tqdm(range(0, len(prepared_texts), self.batch_size), 
                     desc="Encoding batches", unit="batch"):
            batch_texts = prepared_texts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.model.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                if hasattr(outputs, "last_hidden_state"):
                    # Enhanced pooling: use mean pooling of last hidden states
                    attention_mask = inputs['attention_mask']
                    last_hidden = outputs.last_hidden_state
                    
                    # Mean pooling
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    embeddings = embeddings.cpu().numpy()
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs.cpu().numpy()
                else:
                    raise RuntimeError(f"Unexpected model output type: {type(outputs)}")
                
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings
        final_embeddings = final_embeddings / np.linalg.norm(
            final_embeddings, axis=1, keepdims=True
        )
        
        return final_embeddings
    
    def generate_contrastive_pairs(self, texts: List[str], labels: np.ndarray) -> List[ContrastiveExample]:
        """Generate contrastive learning pairs for SetFit training."""
        if not self.use_contrastive:
            return []
        
        contrastive_pairs = []
        label_to_indices = {}
        
        # Group texts by label
        for idx, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        # Generate positive and negative pairs
        for idx, (text, label) in enumerate(zip(texts, labels)):
            # Positive pair: same class
            same_class_indices = [i for i in label_to_indices[label] if i != idx]
            if same_class_indices:
                positive_idx = np.random.choice(same_class_indices)
                positive_text = texts[positive_idx]
            else:
                positive_text = text  # Fallback to self
            
            # Negative pair: different class
            different_labels = [l for l in label_to_indices.keys() if l != label]
            if different_labels:
                negative_label = np.random.choice(different_labels)
                negative_idx = np.random.choice(label_to_indices[negative_label])
                negative_text = texts[negative_idx]
            else:
                negative_text = text  # Fallback to self
            
            contrastive_pairs.append(ContrastiveExample(
                anchor=text,
                positive=positive_text,
                negative=negative_text,
                label=label
            ))
        
        return contrastive_pairs
    
    def train_with_setfit(self, texts: List[str], labels: np.ndarray) -> None:
        """Train embedding model using SetFit contrastive learning."""
        # If SentenceTransformer is available, use the SetFit path
        if self.use_setfit and self.sentence_model is not None:
            logger.info("Starting SetFit contrastive training...")
            
            # Generate contrastive pairs
            contrastive_pairs = self.generate_contrastive_pairs(texts, labels)
            
            if not contrastive_pairs:
                logger.warning("No contrastive pairs generated, skipping SetFit training")
                return
            
            # Prepare training data
            train_texts = []
            train_labels = []
            
            for pair in contrastive_pairs:
                # Positive pairs
                train_texts.extend([pair.anchor, pair.positive])
                train_labels.extend([1, 1])
                
                # Negative pairs
                train_texts.extend([pair.anchor, pair.negative])
                train_labels.extend([0, 0])
            
            # Create contrastive loss
            train_dataset = list(zip(train_texts, train_labels))
            
            # Define contrastive loss
            from sentence_transformers import losses
            from sentence_transformers.util import pytorch_cos_sim
            contrastive_loss = losses.ContrastiveLoss(
                model=self.sentence_model,
                distance_metric=pytorch_cos_sim,
                margin=self.config.model.contrastive_margin
            )
            
            # Train the model
            dataloader = self.sentence_model.get_dataloader(
                train_dataset,
                batch_size=self.config.model.setfit_batch_size
            )
            
            self.sentence_model.fit(
                [(dataloader, contrastive_loss)],
                epochs=self.config.model.setfit_epochs,
                warmup_steps=10,
                output_path=str(Path(self.config.model.model_cache_root) / "setfit_finetuned")
            )
            
            logger.info("SetFit training completed")
            return

        # Fallback: transformers-only QLoRA contrastive fine-tuning
        if not self.use_lora:
            logger.warning("QLoRA is disabled; skipping transformer contrastive training.")
            return
        if not PEFT_AVAILABLE:
            logger.warning("peft is not available; cannot train LoRA adapters.")
            return
        if self.model is None or self.tokenizer is None:
            logger.warning("Transformer model/tokenizer not ready; skip contrastive training.")
            return
        
        logger.info("Starting transformers QLoRA contrastive fine-tuning (fallback)...")
        self.model.train()
        device = self.device
        margin = getattr(self.config.model, 'contrastive_margin', 0.5)
        batch_size = getattr(self.config.model, 'setfit_batch_size', 8)
        epochs = getattr(self.config.model, 'setfit_epochs', 1)
        lr = 1e-4

        # Small training set: subsample for speed/safety
        max_pairs = min(512, len(texts))
        sub_idx = np.random.choice(len(texts), size=max_pairs, replace=False)
        texts_sub = [texts[i] for i in sub_idx]
        labels_sub = labels[sub_idx]

        pairs = self.generate_contrastive_pairs(texts_sub, labels_sub)
        if not pairs:
            logger.warning("No pairs generated; skip transformers contrastive training.")
            self.model.eval()
            return

        # Optimizer (train LoRA adapter params only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not any(p.requires_grad for p in self.model.parameters()):
            # Try to set LoRA params trainable if PEFT wrapped
            for n, p in self.model.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        def mean_pool(last_hidden, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_emb / sum_mask

        # Training loop
        for epoch in range(epochs):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                anchor_texts = [b.anchor for b in batch]
                pos_texts = [b.positive for b in batch]
                neg_texts = [b.negative for b in batch]

                def encode_batch(txts):
                    inputs = self.tokenizer(
                        self._prepare_text_with_instruction(txts),
                        padding=True, truncation=True,
                        max_length=self.config.model.max_seq_length,
                        return_tensors='pt'
                    ).to(device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.last_hidden_state
                    emb = mean_pool(last_hidden, inputs['attention_mask'])
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    return emb

                anchor_emb = encode_batch(anchor_texts)
                pos_emb = encode_batch(pos_texts)
                neg_emb = encode_batch(neg_texts)

                # Margin ranking loss with cosine similarity
                cos = torch.sum(anchor_emb * pos_emb, dim=1)
                cos_neg = torch.sum(anchor_emb * neg_emb, dim=1)
                zeros = torch.zeros_like(cos)
                loss = torch.mean(torch.maximum(zeros, margin - cos + cos_neg))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch)

            avg = total_loss / max(1, len(pairs))
            logger.info(f"Epoch {epoch+1}/{epochs} - contrastive loss: {avg:.4f}")
        
        self.model.eval()
        logger.info("Transformers QLoRA contrastive fine-tuning completed.")
        
        logger.info("Starting SetFit contrastive training...")
        
        # Generate contrastive pairs
        contrastive_pairs = self.generate_contrastive_pairs(texts, labels)
        
        if not contrastive_pairs:
            logger.warning("No contrastive pairs generated, skipping SetFit training")
            return
        
        # Prepare training data
        train_texts = []
        train_labels = []
        
        for pair in contrastive_pairs:
            # Positive pairs
            train_texts.extend([pair.anchor, pair.positive])
            train_labels.extend([1, 1])
            
            # Negative pairs
            train_texts.extend([pair.anchor, pair.negative])
            train_labels.extend([0, 0])
        
        # Create contrastive loss
        train_dataset = list(zip(train_texts, train_labels))
        
        # Define contrastive loss
        contrastive_loss = losses.ContrastiveLoss(
            model=self.sentence_model,
            distance_metric=pytorch_cos_sim,
            margin=self.config.model.contrastive_margin
        )
        
        # Train the model
        dataloader = self.sentence_model.get_dataloader(
            train_dataset,
            batch_size=self.config.model.setfit_batch_size
        )
        
        self.sentence_model.fit(
            [(dataloader, contrastive_loss)],
            epochs=self.config.model.setfit_epochs,
            warmup_steps=100,
            output_path=str(Path(self.config.model.model_cache_root) / "setfit_finetuned")
        )
        
        logger.info("SetFit training completed")
    
    def optimize_threshold(self, val_embeddings: np.ndarray, val_labels: np.ndarray) -> float:
        """Optimize decision threshold for binary classification."""
        if not SKLEARN_AVAILABLE:
            return 0.5
        
        # Get probabilities from a simple classifier
        from sklearn.linear_model import LogisticRegression
        temp_classifier = LogisticRegression(class_weight='balanced')
        temp_classifier.fit(val_embeddings, val_labels)
        val_probs = temp_classifier.predict_proba(val_embeddings)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"Optimized threshold: {optimal_threshold:.4f}")
        return optimal_threshold
    
    def save_embeddings(self, embeddings: np.ndarray, path: Union[str, Path]):
        """Save embeddings to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.npz':
            np.savez_compressed(path, embeddings=embeddings)
        elif path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(embeddings, f)
        else:
            np.save(path, embeddings)
        
        logger.info(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path: Union[str, Path]) -> np.ndarray:
        """Load embeddings from file."""
        path = Path(path)
        
        if path.suffix == '.npz':
            data = np.load(path)
            return data['embeddings']
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return np.load(path)


class EnhancedClassifier:
    """Enhanced classifier with advanced training options."""
    
    def __init__(self, config):
        self.config = config
        self.classifier_type = config.training.classifier_type
        self.classifier = None
        self.is_fitted = False
        self.optimal_threshold = 0.5
        
        self._build_classifier()
    
    def _build_classifier(self):
        """Build enhanced classifier with optimized hyperparameters."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        params = getattr(self.config.training, self.classifier_type, {})
        
        if self.classifier_type == "logistic_regression":
            # Remove solver from params if it exists to avoid conflict
            params = {k: v for k, v in params.items() if k != 'solver'}
            self.classifier = LogisticRegression(
                max_iter=self.config.training.max_iter,
                class_weight=self.config.training.class_weight,
                solver='liblinear',  # Better for small datasets
                **params
            )
        
        elif self.classifier_type == "svm":
            self.classifier = SVC(
                probability=True,
                class_weight=self.config.training.class_weight,
                kernel='rbf',
                **params
            )
        
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                class_weight=self.config.training.class_weight,
                random_state=self.config.data.random_state,
                n_jobs=-1,  # Use all cores
                **params
            )
        
        elif self.classifier_type == "neural_network":
            self.classifier = MLPClassifier(
                max_iter=self.config.training.max_iter,
                random_state=self.config.data.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                **params
            )
        
        elif self.classifier_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost is required for XGBoost classifier")
            
            self.classifier = xgb.XGBClassifier(
                random_state=self.config.data.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                **params
            )
        
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        logger.info(f"Built enhanced {self.classifier_type} classifier")
    
    def fit(self, X: np.ndarray, y: np.ndarray, val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None):
        """Train the classifier with validation for early stopping."""
        logger.info(f"Training enhanced {self.classifier_type} classifier on {len(X)} samples")
        
        # Handle early stopping for XGBoost
        if self.classifier_type == "xgboost" and val_X is not None and val_y is not None:
            self.classifier.fit(
                X, y,
                eval_set=[(val_X, val_y)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.classifier.fit(X, y)
        
        self.is_fitted = True
        
        # Optimize threshold if validation data is available
        if val_X is not None and val_y is not None:
            self._optimize_threshold(val_X, val_y)
        
        logger.info("Enhanced classifier training completed")
    
    def _optimize_threshold(self, val_X: np.ndarray, val_y: np.ndarray):
        """Optimize decision threshold using validation data."""
        if not hasattr(self.classifier, 'predict_proba'):
            return
        
        val_probs = self.classifier.predict_proba(val_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(val_y, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        logger.info(f"Optimized threshold: {self.optimal_threshold:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with optimized threshold."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(X)[:, 1]
            return (probs >= self.optimal_threshold).astype(int)
        else:
            return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            # Fallback for classifiers without probability support
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            probabilities = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0
            return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            return np.abs(self.classifier.coef_).flatten()
        else:
            return None
    
    def save_model(self, path: Union[str, Path]):
        """Save the enhanced classifier."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'optimal_threshold': self.optimal_threshold,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Enhanced classifier saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load a trained enhanced classifier."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        self.config = model_data.get('config', self.config)
        
        self.is_fitted = True
        logger.info(f"Enhanced classifier loaded from {path}")


class EnhancedEnsembleClassifier:
    """Enhanced ensemble classifier with weighted voting and advanced methods."""
    
    def __init__(self, config):
        self.config = config
        self.classifiers = {}
        self.weights = None
        self.is_fitted = False
        
        self._build_ensemble()
    
    def _build_ensemble(self):
        """Build enhanced ensemble of classifiers."""
        methods = self.config.training.ensemble_methods
        
        for method in methods:
            # Create a temporary config for each method
            temp_config = self.config
            temp_config.training.classifier_type = method
            
            try:
                classifier = EnhancedClassifier(temp_config)
                self.classifiers[method] = classifier
                logger.info(f"Added enhanced {method} to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {method} to ensemble: {e}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None):
        """Train all classifiers in the ensemble."""
        logger.info(f"Training enhanced ensemble with {len(self.classifiers)} classifiers")
        
        for name, classifier in self.classifiers.items():
            logger.info(f"Training enhanced {name}")
            classifier.fit(X, y, val_X, val_y)
        
        # Calculate weights based on validation performance
        self._calculate_weights(val_X, val_y)
        self.is_fitted = True
        
        logger.info("Enhanced ensemble training completed")
    
    def _calculate_weights(self, val_X: Optional[np.ndarray], val_y: Optional[np.ndarray]):
        """Calculate ensemble weights based on validation performance."""
        if val_X is None or val_y is None:
            # Equal weights if no validation data
            self.weights = np.ones(len(self.classifiers)) / len(self.classifiers)
            return
        
        # Calculate weights based on F1 scores
        from sklearn.metrics import f1_score
        scores = []
        
        for name, classifier in self.classifiers.items():
            predictions = classifier.predict(val_X)
            score = f1_score(val_y, predictions, average='binary')
            scores.append(score)
            logger.info(f"{name} validation F1: {score:.4f}")
        
        # Convert scores to weights (higher score = higher weight)
        scores = np.array(scores)
        self.weights = scores / scores.sum()
        
        logger.info(f"Ensemble weights: {dict(zip(self.classifiers.keys(), self.weights))}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make enhanced ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get probability predictions from all classifiers
        all_probabilities = []
        for classifier in self.classifiers.values():
            probabilities = classifier.predict_proba(X)
            if probabilities.shape[1] == 2:
                all_probabilities.append(probabilities[:, 1])  # Positive class probability
            else:
                all_probabilities.append(probabilities[:, 0])  # Fallback
        
        # Weighted average of probabilities
        weighted_probabilities = np.zeros(len(X))
        for i, probs in enumerate(all_probabilities):
            weighted_probabilities += self.weights[i] * probs
        
        # Convert to binary predictions using 0.5 threshold
        return (weighted_probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get enhanced ensemble probability predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get probabilities from all classifiers
        all_probabilities = []
        for classifier in self.classifiers.values():
            probabilities = classifier.predict_proba(X)
            all_probabilities.append(probabilities)
        
        # Weighted average of probabilities
        all_probabilities = np.array(all_probabilities)
        weighted_probabilities = np.zeros_like(all_probabilities[0])
        
        for i, probabilities in enumerate(all_probabilities):
            weighted_probabilities += self.weights[i] * probabilities
        
        return weighted_probabilities


class EnhancedModelManager:
    """Enhanced model manager with advanced training capabilities."""
    
    def __init__(self, config, embedder: Optional[EnhancedQwenEmbedder] = None):
        self.config = config
        self.embedder = embedder or EnhancedQwenEmbedder(config)
        
        if config.training.use_ensemble:
            self.classifier = EnhancedEnsembleClassifier(config)
        else:
            self.classifier = EnhancedClassifier(config)
        
        self.embeddings_cache = {}
    
    def prepare_embeddings(self, 
                          train_texts: List[str], 
                          val_texts: Optional[List[str]] = None,
                          test_texts: Optional[List[str]] = None,
                          cache_dir: Optional[Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare embeddings with advanced features."""
        logger.info("Preparing enhanced embeddings...")
        
        # Train embeddings
        train_embeddings = self.embedder.encode(train_texts)
        logger.info(f"Generated train embeddings: {train_embeddings.shape}")
        
        # Validation embeddings
        val_embeddings = None
        if val_texts:
            val_embeddings = self.embedder.encode(val_texts)
            logger.info(f"Generated validation embeddings: {val_embeddings.shape}")
        
        # Test embeddings
        test_embeddings = None
        if test_texts:
            test_embeddings = self.embedder.encode(test_texts)
            logger.info(f"Generated test embeddings: {test_embeddings.shape}")
        
        # Cache embeddings if configured
        if self.config.save_embeddings and cache_dir:
            self._cache_embeddings(train_embeddings, val_embeddings, test_embeddings, cache_dir)
        
        return train_embeddings, val_embeddings, test_embeddings
    
    def train_with_setfit(self, train_texts: List[str], train_labels: np.ndarray):
        """Train embedding model using SetFit before classifier training."""
        if self.embedder.use_setfit or self.embedder.use_contrastive:
            # Will use SetFit if sentence_model exists; otherwise fallback to transformers contrastive QLoRA
            self.embedder.train_with_setfit(train_texts, train_labels)
    
    def train_classifier(self, 
                        train_embeddings: np.ndarray, 
                        train_labels: np.ndarray,
                        val_embeddings: Optional[np.ndarray] = None,
                        val_labels: Optional[np.ndarray] = None):
        """Train the enhanced classifier."""
        self.classifier.fit(train_embeddings, train_labels, val_embeddings, val_labels)
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Make enhanced predictions."""
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Get enhanced probability predictions."""
        return self.classifier.predict_proba(embeddings)
    
    def _cache_embeddings(self, 
                          train_embeddings: np.ndarray,
                          val_embeddings: Optional[np.ndarray],
                          test_embeddings: Optional[np.ndarray],
                          cache_dir: Path):
        """Cache embeddings to disk."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedder.save_embeddings(train_embeddings, cache_dir / "train_embeddings.npz")
        
        if val_embeddings is not None:
            self.embedder.save_embeddings(val_embeddings, cache_dir / "val_embeddings.npz")
        
        if test_embeddings is not None:
            self.embedder.save_embeddings(test_embeddings, cache_dir / "test_embeddings.npz")
        
        logger.info(f"Enhanced embeddings cached to {cache_dir}")
    
    def save_models(self, save_dir: Path):
        """Save all enhanced models."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        if hasattr(self.classifier, 'save_model'):
            self.classifier.save_model(save_dir / "enhanced_classifier.pkl")
        else:
            # Fallback for ensemble
            with open(save_dir / "enhanced_classifier.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
        
        # Save LoRA adapters if available
        try:
            if PEFT_AVAILABLE and hasattr(self.embedder, 'model') and hasattr(self.embedder.model, 'save_pretrained'):
                adapter_dir = save_dir / "lora_adapter"
                self.embedder.model.save_pretrained(adapter_dir)
                logger.info(f"Saved LoRA adapter to {adapter_dir}")
        except Exception as e:
            logger.warning(f"Failed to save LoRA adapter: {e}")

        logger.info(f"Enhanced models saved to {save_dir}")
    
    def load_models(self, load_dir: Path):
        """Load all enhanced models."""
        # Load classifier
        if hasattr(self.classifier, 'load_model'):
            self.classifier.load_model(load_dir / "enhanced_classifier.pkl")
        else:
            # Fallback for ensemble
            with open(load_dir / "enhanced_classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
        
        logger.info(f"Enhanced models loaded from {load_dir}")
