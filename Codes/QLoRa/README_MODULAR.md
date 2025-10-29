# Advanced Pseudo-Civility Detection - Modular Implementation

## Overview

This modular implementation provides a comprehensive, production-ready pipeline for pseudo-civility detection using advanced embedding models and machine learning techniques. The system is designed with modularity, extensibility, and reproducibility in mind.

## Architecture

### Core Components

1. **Configuration Management** (`config.py`)
   - Centralized configuration with JSON/YAML support
   - Predefined configurations for different scenarios
   - Validation and type safety

2. **Data Processing** (`data_processor.py`)
   - Multi-dataset loading and preprocessing
   - Advanced text cleaning and normalization
   - Data splitting and balancing
   - Support for various data formats

3. **Model Management** (`models.py`)
   - Qwen3-Embedding integration with advanced features
   - Multiple classifier implementations (Logistic Regression, SVM, Random Forest, XGBoost, Neural Networks)
   - Ensemble methods and cross-validation
   - Model persistence and loading

4. **Evaluation** (`evaluation.py`)
   - Comprehensive metrics calculation
   - Advanced error analysis
   - Visualization generation
   - Calibration analysis

5. **Main Orchestration** (`main.py`)
   - Complete pipeline execution
   - Command-line interface
   - Experiment tracking
   - Report generation

## Key Features

### Advanced Embedding Support
- **Qwen3-Embedding Integration**: State-of-the-art embedding model
- **Instruction-based Embeddings**: Task-specific prompt engineering
- **Contrastive Learning**: Optional contrastive fine-tuning
- **LoRA Support**: Parameter-efficient fine-tuning
- **Quantization**: Memory-efficient inference

### Multiple Classifier Types
- **Logistic Regression**: Baseline linear classifier
- **SVM**: Support Vector Machine with various kernels
- **Random Forest**: Ensemble tree-based method
- **XGBoost**: Gradient boosting with advanced features
- **Neural Networks**: Custom MLP architectures
- **Ensemble Methods**: Voting and stacking ensembles

### Comprehensive Evaluation
- **Basic Metrics**: Accuracy, F1, Precision, Recall
- **Probabilistic Metrics**: ROC-AUC, PR-AUC
- **Calibration Analysis**: Expected Calibration Error
- **Error Analysis**: Detailed error pattern analysis
- **Visualization**: Confusion matrices, ROC curves, calibration plots

### Data Processing Features
- **Multi-dataset Support**: ChnSentiCorp, Civil Comments, Wikipedia Politeness, GoEmotions, ToxiGen
- **Advanced Preprocessing**: Text cleaning, deduplication, length filtering
- **Data Balancing**: Class imbalance handling
- **Metadata Preservation**: Source tracking and analysis

## Installation

### Requirements
```bash
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn
pip install xgboost  # Optional, for XGBoost classifier
```

### Optional Dependencies
```bash
pip install peft  # For LoRA fine-tuning
pip install bitsandbytes  # For quantization
```

## Usage

### Basic Usage

1. **Quick Test with Default Configuration**
```bash
python main.py --quick-test
```

2. **Binary Classification**
```bash
python main.py --config-type binary --datasets chnsenticorp civil_comments
```

3. **Ensemble Classification**
```bash
python main.py --config-type ensemble --classifier xgboost --ensemble
```

4. **Custom Configuration**
```bash
python main.py --config configs/binary_config.json
```

### Advanced Usage

1. **Cross-Validation**
```bash
python main.py --datasets chnsenticorp --cross-validation --cv-folds 5
```

2. **Specific Classifier**
```bash
python main.py --classifier xgboost --max-samples 5000
```

3. **Custom Output Directory**
```bash
python main.py --output-dir experiments/civility_detection --experiment-name run_001
```

### Configuration Examples

#### Binary Classification Configuration
```json
{
  "experiment_name": "binary_civility_detection",
  "data": {
    "datasets": ["chnsenticorp", "civil_comments"],
    "max_samples": 1000,
    "balance_classes": true
  },
  "model": {
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "use_instruction_prompt": true,
    "instruction": "Represent this comment for civility classification."
  },
  "training": {
    "classifier_type": "logistic_regression",
    "class_weight": "balanced"
  },
  "evaluation": {
    "metrics": ["accuracy", "f1", "roc_auc"],
    "calibration": true,
    "error_analysis": true
  }
}
```

#### Ensemble Configuration
```json
{
  "experiment_name": "ensemble_civility_detection",
  "training": {
    "use_ensemble": true,
    "ensemble_methods": ["logistic_regression", "svm", "random_forest", "xgboost"]
  },
  "model": {
    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 64
  }
}
```

## Output Structure

```
outputs/
├── experiment_name/
│   ├── config.json                 # Experiment configuration
│   ├── experiment.log              # Detailed log file
│   ├── experiment_report.md        # Human-readable report
│   ├── embeddings/                 # Cached embeddings (if saved)
│   ├── models/                     # Trained models (if saved)
│   ├── validation/                 # Validation results
│   │   ├── evaluation_results.json
│   │   ├── evaluation_summary.md
│   │   ├── confusion_matrix.png
│   │   └── calibration_curve.png
│   └── test/                       # Test results
│       ├── evaluation_results.json
│       ├── evaluation_summary.md
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── precision_recall_curve.png
```

## Advanced Features

### Instruction-based Embeddings
The system supports instruction-based embeddings to improve task-specific performance:

```python
instruction = "Represent this comment for civility classification. Return a vector that captures politeness, sentiment and toxicity."
embeddings = model.encode(texts, instruction=instruction)
```

### Contrastive Learning
Optional contrastive fine-tuning for better representation learning:

```python
# Configuration
{
  "model": {
    "use_contrastive": true,
    "contrastive_temperature": 0.07,
    "contrastive_margin": 0.5
  }
}
```

### LoRA Fine-tuning
Parameter-efficient fine-tuning for large models:

```python
# Configuration
{
  "model": {
    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 64,
    "lora_dropout": 0.1
  }
}
```

### Ensemble Methods
Multiple classifier combination for improved performance:

```python
# Configuration
{
  "training": {
    "use_ensemble": true,
    "ensemble_methods": ["logistic_regression", "svm", "random_forest", "xgboost"]
  }
}
```

## Evaluation Metrics

### Basic Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity or true positive rate

### Probabilistic Metrics
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures how well predicted probabilities match actual frequencies

### Error Analysis
- **Confusion Matrix**: Detailed classification breakdown
- **Error Patterns**: Systematic error identification
- **Confidence Analysis**: Prediction confidence vs accuracy

## Performance Optimization

### Memory Optimization
- **Batch Processing**: Configurable batch sizes for embedding generation
- **Gradient Checkpointing**: Memory-efficient training
- **Quantization**: 8-bit quantization for reduced memory usage

### Speed Optimization
- **Parallel Processing**: Multi-threaded data processing
- **Caching**: Embedding and model caching
- **Early Stopping**: Prevent overfitting and reduce training time

## Extensibility

### Adding New Datasets
```python
class CustomDatasetLoader(BaseDatasetLoader):
    def load_data(self, file_path: str) -> List[DataExample]:
        # Custom loading logic
        pass
```

### Adding New Classifiers
```python
class CustomClassifier(BaseClassifier):
    def train(self, X: np.ndarray, y: np.ndarray):
        # Custom training logic
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Custom prediction logic
        pass
```

### Adding New Metrics
```python
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, y_true, y_pred):
        # Custom metric calculation
        pass
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size: `--batch-size 8`
   - Enable quantization: Set `"quantize": true` in config
   - Use smaller model: Change `model_name` to smaller variant

2. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Data Loading Issues**
   - Verify data paths in configuration
   - Check data format compatibility
   - Ensure sufficient disk space

### Debug Mode
```bash
python main.py --log-level DEBUG --quick-test
```

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

### Testing
```bash
python test_modular.py  # Run modular tests
python main.py --quick-test  # Run integration test
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{pseudo_civility_detection,
  title={Advanced Pseudo-Civility Detection with Qwen3-Embedding},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Acknowledgments

- Qwen team for the Qwen3-Embedding model
- Hugging Face for the transformers library
- Scikit-learn for machine learning utilities
- The open-source community for various tools and libraries
