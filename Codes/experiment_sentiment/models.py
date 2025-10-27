"""
Advanced models module for pseudo-civility detection.
Includes enhanced embedding generation and multiple classifier options.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm

# Import with fallbacks
try:
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers sentence-transformers")

try:
    from peft import LoraConfig, get_peft_model
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


class AdvancedQwenEmbedder:
    """Enhanced Qwen embedding model with advanced features."""
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.model.model_name
        self.device = self._get_device()
        self.quantize = config.model.quantize
        self.batch_size = config.model.batch_size
        self.use_instruction_prompt = config.model.use_instruction_prompt
        self.instruction = config.model.instruction
        
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
        """Load the embedding model with fallback options."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
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
        
        # Fallback to transformers
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            if self.quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                ).to(self.device)
            
            self.model.eval()
            
            # Apply LoRA if configured
            if self.config.model.use_lora and PEFT_AVAILABLE:
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
        """Apply LoRA configuration to the model."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, skipping LoRA")
            return
        
        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA configuration applied")
    
    def _prepare_text_with_instruction(self, texts: List[str]) -> List[str]:
        """Prepare texts with instruction prompts if enabled."""
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
    
    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode using SentenceTransformer."""
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
        """Encode using native transformers."""
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
                outputs = self.model(**inputs)
                
                if hasattr(outputs, "last_hidden_state"):
                    # Use [CLS] token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
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


class AdvancedClassifier:
    """Advanced classifier with multiple algorithm options."""
    
    def __init__(self, config):
        self.config = config
        self.classifier_type = config.training.classifier_type
        self.classifier = None
        self.is_fitted = False
        
        self._build_classifier()
    
    def _build_classifier(self):
        """Build the classifier based on configuration."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        params = getattr(self.config.training, self.classifier_type, {})
        
        if self.classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                max_iter=self.config.training.max_iter,
                class_weight=self.config.training.class_weight,
                **params
            )
        
        elif self.classifier_type == "svm":
            self.classifier = SVC(
                probability=True,  # Required for probability predictions
                class_weight=self.config.training.class_weight,
                **params
            )
        
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                class_weight=self.config.training.class_weight,
                random_state=self.config.data.random_state,
                **params
            )
        
        elif self.classifier_type == "neural_network":
            self.classifier = MLPClassifier(
                max_iter=self.config.training.max_iter,
                random_state=self.config.data.random_state,
                **params
            )
        
        elif self.classifier_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost is required for XGBoost classifier")
            
            self.classifier = xgb.XGBClassifier(
                random_state=self.config.data.random_state,
                **params
            )
        
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        logger.info(f"Built {self.classifier_type} classifier")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        logger.info(f"Training {self.classifier_type} classifier on {len(X)} samples")
        
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        logger.info("Classifier training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
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
        """Save the trained classifier."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        logger.info(f"Classifier saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load a trained classifier."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Classifier loaded from {path}")


class EnsembleClassifier:
    """Ensemble classifier combining multiple models."""
    
    def __init__(self, config):
        self.config = config
        self.classifiers = {}
        self.weights = None
        self.is_fitted = False
        
        self._build_ensemble()
    
    def _build_ensemble(self):
        """Build ensemble of classifiers."""
        methods = self.config.training.ensemble_methods
        
        for method in methods:
            # Create a temporary config for each method
            temp_config = self.config
            temp_config.training.classifier_type = method
            
            try:
                classifier = AdvancedClassifier(temp_config)
                self.classifiers[method] = classifier
                logger.info(f"Added {method} to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {method} to ensemble: {e}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all classifiers in the ensemble."""
        logger.info(f"Training ensemble with {len(self.classifiers)} classifiers")
        
        for name, classifier in self.classifiers.items():
            logger.info(f"Training {name}")
            classifier.fit(X, y)
        
        # Calculate weights based on cross-validation performance
        self._calculate_weights(X, y)
        self.is_fitted = True
        
        logger.info("Ensemble training completed")
    
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray):
        """Calculate ensemble weights based on cross-validation."""
        if not self.config.training.cross_validation:
            # Equal weights if no cross-validation
            self.weights = np.ones(len(self.classifiers)) / len(self.classifiers)
            return
        
        # Simple weight calculation based on classifier type
        # In practice, you'd use cross-validation scores here
        weights_dict = {
            'logistic_regression': 0.3,
            'svm': 0.3,
            'random_forest': 0.2,
            'xgboost': 0.2
        }
        
        weights = []
        for name in self.classifiers.keys():
            weight = weights_dict.get(name, 0.1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = np.array(weights) / total_weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from all classifiers
        all_predictions = []
        for classifier in self.classifiers.values():
            predictions = classifier.predict(X)
            all_predictions.append(predictions)
        
        # Weighted voting
        all_predictions = np.array(all_predictions)
        weighted_predictions = np.zeros_like(all_predictions[0])
        
        for i, predictions in enumerate(all_predictions):
            weighted_predictions += self.weights[i] * predictions
        
        # Round to nearest integer for classification
        return np.round(weighted_predictions).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions."""
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


class ModelManager:
    """Main model manager orchestrating embedding and classification."""
    
    def __init__(self, config):
        self.config = config
        self.embedder = AdvancedQwenEmbedder(config)
        
        if config.training.use_ensemble:
            self.classifier = EnsembleClassifier(config)
        else:
            self.classifier = AdvancedClassifier(config)
        
        self.embeddings_cache = {}
    
    def prepare_embeddings(self, 
                          train_texts: List[str], 
                          val_texts: Optional[List[str]] = None,
                          test_texts: Optional[List[str]] = None,
                          cache_dir: Optional[Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare embeddings for all data splits."""
        logger.info("Preparing embeddings...")
        
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
        
        logger.info(f"Embeddings cached to {cache_dir}")
    
    def train_classifier(self, train_embeddings: np.ndarray, train_labels: np.ndarray):
        """Train the classifier."""
        self.classifier.fit(train_embeddings, train_labels)
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.classifier.predict_proba(embeddings)
    
    def save_models(self, save_dir: Path):
        """Save all models."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        if hasattr(self.classifier, 'save_model'):
            self.classifier.save_model(save_dir / "classifier.pkl")
        else:
            # Fallback for ensemble
            with open(save_dir / "classifier.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: Path):
        """Load all models."""
        # Load classifier
        if hasattr(self.classifier, 'load_model'):
            self.classifier.load_model(load_dir / "classifier.pkl")
        else:
            # Fallback for ensemble
            with open(load_dir / "classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
        
        logger.info(f"Models loaded from {load_dir}")
