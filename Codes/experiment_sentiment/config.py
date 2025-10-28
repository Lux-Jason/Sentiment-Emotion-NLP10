"""
Configuration management for the pseudo-civility detection system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import json


@dataclass
class DataConfig:
    """Data processing configuration."""
    datasets: List[str] = field(default_factory=lambda: ["chnsenticorp", "wikipedia_politeness", "go_emotions", "civil_comments", "toxigen"])
    max_samples: Optional[int] = None
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    archive_root: str = "Archive"
    output_dir: str = "outputs"
    
    # Text preprocessing
    min_text_length: int = 10
    max_text_length: int = 512
    remove_duplicates: bool = True
    balance_classes: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    model_cache_root: str = "model_cache"
    device: Optional[str] = None
    quantize: bool = True
    batch_size: int = 16
    max_seq_length: int = 512
    
    # Advanced features
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Embedding enhancements
    use_instruction_prompt: bool = True
    instruction: str = "Represent this comment for civility classification. Return a vector that captures politeness, sentiment and toxicity."
    
    # Contrastive learning
    use_contrastive: bool = False
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    
    # SetFit training
    use_setfit: bool = False
    setfit_epochs: int = 3
    setfit_batch_size: int = 16


@dataclass
class TrainingConfig:
    """Training configuration."""
    classifier_type: str = "logistic_regression"  # Options: logistic_regression, svm, random_forest, xgboost, neural_network
    max_iter: int = 1000
    class_weight: str = "balanced"
    
    # Hyperparameters for different classifiers
    logistic_regression: Dict = field(default_factory=lambda: {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs"
    })
    
    svm: Dict = field(default_factory=lambda: {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    })
    
    random_forest: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    })
    
    xgboost: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    })
    
    neural_network: Dict = field(default_factory=lambda: {
        "hidden_layer_sizes": [256, 128],
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate": "adaptive"
    })
    
    # Advanced training options
    use_ensemble: bool = False
    ensemble_methods: List[str] = field(default_factory=lambda: ["logistic_regression", "svm", "random_forest"])
    cross_validation: bool = True
    cv_folds: int = 5
    threshold_optimization: bool = True
    early_stopping: bool = True
    learning_rate_schedule: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall", "roc_auc", "confusion_matrix"])
    average: str = "macro"  # For multi-class metrics
    save_predictions: bool = True
    save_probabilities: bool = True
    threshold_optimization: bool = True
    
    # Advanced evaluation
    calibration: bool = True
    feature_importance: bool = True
    error_analysis: bool = True
    macro_averaging: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    experiment_name: str = "pseudo_civility_detection"
    description: str = "Advanced pseudo-civility detection with modular architecture"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Logging and saving
    log_level: str = "INFO"
    save_embeddings: bool = True
    save_models: bool = True
    save_results: bool = True
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle non-serializable objects
        config_dict = {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "log_level": self.log_level,
            "save_embeddings": self.save_embeddings,
            "save_models": self.save_models,
            "save_results": self.save_results
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Create nested config objects
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            experiment_name=config_dict.get("experiment_name", "pseudo_civility_detection"),
            description=config_dict.get("description", ""),
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            log_level=config_dict.get("log_level", "INFO"),
            save_embeddings=config_dict.get("save_embeddings", True),
            save_models=config_dict.get("save_models", True),
            save_results=config_dict.get("save_results", True)
        )


# Default configurations for different experiment types
def get_default_config_binary() -> ExperimentConfig:
    """Get default configuration for binary classification."""
    config = ExperimentConfig()
    config.experiment_name = "binary_civility_detection"
    config.training.classifier_type = "logistic_regression"
    config.evaluation.average = "binary"
    return config


def get_default_config_multiclass() -> ExperimentConfig:
    """Get default configuration for multi-class classification."""
    config = ExperimentConfig()
    config.experiment_name = "multiclass_civility_detection"
    config.training.classifier_type = "xgboost"
    config.evaluation.average = "macro"
    return config


def get_default_config_ensemble() -> ExperimentConfig:
    """Get default configuration for ensemble learning."""
    config = ExperimentConfig()
    config.experiment_name = "ensemble_civility_detection"
    config.training.use_ensemble = True
    config.training.cross_validation = True
    config.evaluation.calibration = True
    return config
