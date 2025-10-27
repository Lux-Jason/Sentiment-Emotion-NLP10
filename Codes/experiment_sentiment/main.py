#!/usr/bin/env python3
"""
Main orchestration script for advanced pseudo-civility detection.

This script provides a complete pipeline for:
- Loading and preprocessing multiple datasets
- Training advanced embedding models with Qwen3-Embedding
- Training multiple classifier types with ensemble support
- Comprehensive evaluation with visualization
- Model persistence and experiment tracking

Usage:
  python main.py --config configs/binary_config.yaml
  python main.py --config configs/ensemble_config.yaml --quick-test
  python main.py --datasets chnsenticorp civil_comments --classifier xgboost
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Codes.experiment_sentiment.config import ExperimentConfig, get_default_config_binary, get_default_config_ensemble
from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample
from Codes.experiment_sentiment.models import ModelManager
from Codes.experiment_sentiment.evaluation import Evaluator


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Pseudo-Civility Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--config-type', type=str, 
                       choices=['binary', 'multiclass', 'ensemble'],
                       default='binary', help='Predefined configuration type')
    
    # Data settings
    parser.add_argument('--datasets', nargs='+', 
                       choices=['chnsenticorp', 'wikipedia_politeness', 'go_emotions', 'civil_comments', 'toxigen'],
                       help='Datasets to use')
    parser.add_argument('--max-samples', type=int, help='Maximum samples per dataset')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    
    # Model settings
    parser.add_argument('--classifier', type=str,
                       choices=['logistic_regression', 'svm', 'random_forest', 'xgboost', 'neural_network'],
                       help='Classifier type')
    parser.add_argument('--model-name', type=str, help='Embedding model name')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, help='Batch size for embedding')
    
    # Training settings
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble classifier')
    parser.add_argument('--cross-validation', action='store_true', help='Use cross-validation')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--save-models', action='store_true', default=True, help='Save trained models')
    parser.add_argument('--save-embeddings', action='store_true', default=True, help='Save embeddings')
    
    # Execution settings
    parser.add_argument('--quick-test', action='store_true', help='Quick test with reduced data')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--resume', action='store_true', help='Resume from existing embeddings')
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create configuration from command line arguments."""
    # Start with predefined config
    if args.config_type == 'binary':
        config = get_default_config_binary()
    elif args.config_type == 'ensemble':
        config = get_default_config_ensemble()
    else:
        config = ExperimentConfig()
    
    # Override with config file if provided
    if args.config and Path(args.config).exists():
        config = ExperimentConfig.load_from_file(args.config)
    
    # Override with command line arguments
    if args.datasets:
        config.data.datasets = args.datasets
    
    if args.max_samples:
        config.data.max_samples = args.max_samples
    
    if args.data_dir:
        config.data.archive_root = args.data_dir
    
    if args.classifier:
        config.training.classifier_type = args.classifier
    
    if args.model_name:
        config.model.model_name = args.model_name
    
    if args.device:
        config.model.device = args.device
    
    if args.batch_size:
        config.model.batch_size = args.batch_size
    
    if args.ensemble:
        config.training.use_ensemble = True
    
    if args.cross_validation:
        config.training.cross_validation = True
    
    if args.output_dir:
        config.data.output_dir = args.output_dir
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if not args.save_models:
        config.save_models = False
    
    if not args.save_embeddings:
        config.save_embeddings = False
    
    if args.quick_test:
        config.data.max_samples = min(config.data.max_samples or 1000, 500)
        config.model.batch_size = min(config.model.batch_size, 8)
        config.training.cross_validation = False
    
    config.log_level = args.log_level
    
    return config


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete experiment pipeline."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Setup output directory
    output_dir = Path(config.data.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging file
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Save configuration
    config.save_to_file(output_dir / "config.json")
    
    experiment_results = {
        'config': config,
        'output_dir': output_dir,
        'start_time': time.time()
    }
    
    try:
        # Step 1: Data Processing
        logger.info("=" * 50)
        logger.info("STEP 1: Data Processing")
        logger.info("=" * 50)
        
        data_processor = DataProcessor(config)
        train_examples, val_examples, test_examples = data_processor.load_and_process_datasets()
        
        # Get data statistics
        data_stats = data_processor.get_data_statistics(train_examples, val_examples, test_examples)
        logger.info(f"Data statistics: {data_stats}")
        
        experiment_results['data_stats'] = data_stats
        
        # Extract texts and labels
        train_texts = [ex.text for ex in train_examples]
        train_labels = np.array([ex.label for ex in train_examples])
        
        val_texts = [ex.text for ex in val_examples] if val_examples else []
        val_labels = np.array([ex.label for ex in val_examples]) if val_examples else np.array([])
        
        test_texts = [ex.text for ex in test_examples] if test_examples else []
        test_labels = np.array([ex.label for ex in test_examples]) if test_examples else np.array([])
        
        # Step 2: Model Training
        logger.info("=" * 50)
        logger.info("STEP 2: Model Training")
        logger.info("=" * 50)
        
        model_manager = ModelManager(config)
        
        # Prepare embeddings
        embeddings_cache_dir = output_dir / "embeddings" if config.save_embeddings else None
        train_embeddings, val_embeddings, test_embeddings = model_manager.prepare_embeddings(
            train_texts, val_texts, test_texts, embeddings_cache_dir
        )
        
        # Train classifier
        model_manager.train_classifier(train_embeddings, train_labels)
        
        # Step 3: Evaluation
        logger.info("=" * 50)
        logger.info("STEP 3: Evaluation")
        logger.info("=" * 50)
        
        evaluator = Evaluator(config)
        
        # Evaluate on validation set
        val_results = {}
        if len(val_examples) > 0:
            val_predictions = model_manager.predict(val_embeddings)
            val_probabilities = model_manager.predict_proba(val_embeddings)
            
            val_results = evaluator.evaluate(
                val_labels, val_predictions, val_probabilities,
                val_texts, [ex.metadata for ex in val_examples],
                output_dir / "validation"
            )
        
        # Evaluate on test set
        test_results = {}
        if len(test_examples) > 0:
            test_predictions = model_manager.predict(test_embeddings)
            test_probabilities = model_manager.predict_proba(test_embeddings)
            
            test_results = evaluator.evaluate(
                test_labels, test_predictions, test_probabilities,
                test_texts, [ex.metadata for ex in test_examples],
                output_dir / "test"
            )
        
        experiment_results['val_results'] = val_results
        experiment_results['test_results'] = test_results
        
        # Step 4: Save Models
        if config.save_models:
            logger.info("=" * 50)
            logger.info("STEP 4: Saving Models")
            logger.info("=" * 50)
            
            model_manager.save_models(output_dir / "models")
        
        # Generate final report
        logger.info("=" * 50)
        logger.info("STEP 5: Final Report")
        logger.info("=" * 50)
        
        generate_final_report(experiment_results, output_dir)
        
        experiment_results['success'] = True
        experiment_results['end_time'] = time.time()
        experiment_results['duration'] = experiment_results['end_time'] - experiment_results['start_time']
        
        logger.info(f"Experiment completed successfully in {experiment_results['duration']:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        experiment_results['success'] = False
        experiment_results['error'] = str(e)
        experiment_results['end_time'] = time.time()
        experiment_results['duration'] = experiment_results['end_time'] - experiment_results['start_time']
        raise
    
    return experiment_results


def generate_final_report(results: dict, output_dir: Path):
    """Generate a comprehensive final report."""
    report_lines = [
        "# Pseudo-Civility Detection Experiment Report\n",
        f"## Experiment: {results['config'].experiment_name}",
        f"**Duration:** {results.get('duration', 0):.2f} seconds",
        f"**Status:** {'✅ Success' if results.get('success', False) else '❌ Failed'}\n",
        "## Configuration\n",
        f"- **Datasets:** {', '.join(results['config'].data.datasets)}",
        f"- **Classifier:** {results['config'].training.classifier_type}",
        f"- **Embedding Model:** {results['config'].model.model_name}",
        f"- **Device:** {results['config'].model.device or 'auto'}",
        f"- **Batch Size:** {results['config'].model.batch_size}",
        f"- **Max Samples:** {results['config'].data.max_samples or 'All'}\n"
    ]
    
    # Data statistics
    if 'data_stats' in results:
        stats = results['data_stats']
        report_lines.extend([
            "## Data Statistics\n",
            f"- **Total Samples:** {stats['total']['count']}",
            f"- **Training:** {stats['train']['count']}",
            f"- **Validation:** {stats['val']['count']}",
            f"- **Test:** {stats['test']['count']}\n"
        ])
    
    # Validation results
    if results.get('val_results'):
        val_results = results['val_results']
        report_lines.append("## Validation Results\n")
        
        if val_results.get('basic_metrics'):
            report_lines.append("### Basic Metrics\n")
            for metric, value in val_results['basic_metrics'].items():
                report_lines.append(f"- **{metric}:** {value:.4f}")
            report_lines.append("")
        
        if val_results.get('probabilistic_metrics'):
            report_lines.append("### Probabilistic Metrics\n")
            for metric, value in val_results['probabilistic_metrics'].items():
                report_lines.append(f"- **{metric}:** {value:.4f}")
            report_lines.append("")
    
    # Test results
    if results.get('test_results'):
        test_results = results['test_results']
        report_lines.append("## Test Results\n")
        
        if test_results.get('basic_metrics'):
            report_lines.append("### Basic Metrics\n")
            for metric, value in test_results['basic_metrics'].items():
                report_lines.append(f"- **{metric}:** {value:.4f}")
            report_lines.append("")
        
        if test_results.get('probabilistic_metrics'):
            report_lines.append("### Probabilistic Metrics\n")
            for metric, value in test_results['probabilistic_metrics'].items():
                report_lines.append(f"- **{metric}:** {value:.4f}")
            report_lines.append("")
        
        if test_results.get('error_analysis'):
            error_patterns = test_results['error_analysis'].get('error_patterns', {})
            report_lines.extend([
                "### Error Analysis\n",
                f"- **Total Errors:** {error_patterns.get('total_errors', 0)}",
                f"- **Error Rate:** {error_patterns.get('error_rate', 0):.4f}\n"
            ])
    
    # Error information
    if 'error' in results:
        report_lines.extend([
            "## Error Information\n",
            f"```\n{results['error']}\n```\n"
        ])
    
    # File structure
    report_lines.extend([
        "## Generated Files\n",
        f"- **Configuration:** `config.json`",
        f"- **Log File:** `experiment.log`",
        f"- **Validation Results:** `validation/`",
        f"- **Test Results:** `test/`",
        f"- **Models:** `models/` (if saved)",
        f"- **Embeddings:** `embeddings/` (if saved)\n"
    ])
    
    # Write report
    with open(output_dir / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Final report generated: {output_dir / 'experiment_report.md'}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Run experiment
        results = run_experiment(config)
        
        if results.get('success', False):
            logger.info("🎉 Experiment completed successfully!")
            return 0
        else:
            logger.error("❌ Experiment failed!")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np  # Import here to avoid circular imports
    sys.exit(main())
