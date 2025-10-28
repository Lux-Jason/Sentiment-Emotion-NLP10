#!/usr/bin/env python3
"""
Enhanced main orchestration script with beautiful visualizations and automatic versioning.
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
from Codes.experiment_sentiment.enhanced_visualizer import EnhancedVisualizer


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
        description="Enhanced Pseudo-Civility Detection with Beautiful Visualizations",
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
    
    # Visualization settings
    parser.add_argument('--enhanced-viz', action='store_true', default=True, help='Use enhanced visualizations')
    parser.add_argument('--interactive-dashboard', action='store_true', help='Create interactive dashboard')
    parser.add_argument('--embedding-viz', action='store_true', help='Create embedding visualizations')
    
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


def run_enhanced_experiment(config: ExperimentConfig, args) -> dict:
    """Run enhanced experiment pipeline with beautiful visualizations."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting enhanced experiment: {config.experiment_name}")
    logger.info(f"Configuration: {config}")
    
    # Initialize enhanced visualizer
    visualizer = EnhancedVisualizer()
    
    # Create results directory with timestamp
    results_dir = visualizer.create_results_directory(
        Path(config.data.output_dir), 
        config.experiment_name
    )
    
    # Setup logging file
    log_file = results_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Save configuration
    config.save_to_file(results_dir / "config.json")
    
    experiment_results = {
        'config': config,
        'results_dir': results_dir,
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
        embeddings_cache_dir = results_dir / "embeddings" if config.save_embeddings else None
        train_embeddings, val_embeddings, test_embeddings = model_manager.prepare_embeddings(
            train_texts, val_texts, test_texts, embeddings_cache_dir
        )
        
        # Train classifier
        model_manager.train_classifier(train_embeddings, train_labels)
        
        # Step 3: Enhanced Evaluation
        logger.info("=" * 50)
        logger.info("STEP 3: Enhanced Evaluation")
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
                results_dir / "validation"
            )
        
        # Evaluate on test set
        test_results = {}
        if len(test_examples) > 0:
            test_predictions = model_manager.predict(test_embeddings)
            test_probabilities = model_manager.predict_proba(test_embeddings)
            
            test_results = evaluator.evaluate(
                test_labels, test_predictions, test_probabilities,
                test_texts, [ex.metadata for ex in test_examples],
                results_dir / "test"
            )
        
        experiment_results['val_results'] = val_results
        experiment_results['test_results'] = test_results
        
        # Step 4: Enhanced Visualizations
        logger.info("=" * 50)
        logger.info("STEP 4: Enhanced Visualizations")
        logger.info("=" * 50)
        
        # Create enhanced visualizations
        charts_dir = results_dir / "charts"
        
        # Enhanced confusion matrix
        if test_results and 'error_analysis' in test_results:
            cm_data = test_results['error_analysis']['confusion_matrix']['matrix']
            cm = np.array(cm_data)
            
            visualizer.plot_enhanced_confusion_matrix(
                cm, 
                save_path=charts_dir / "enhanced_confusion_matrix.png"
            )
            plt.close('all')
        
        # Enhanced ROC curve
        if test_results and 'probabilistic_metrics' in test_results:
            # This would need actual ROC data - for now we'll create a placeholder
            # In a real implementation, you'd extract this from the evaluation results
            pass
        
        # Metrics comparison
        if val_results and test_results:
            metrics_comparison = {
                'Validation': val_results.get('basic_metrics', {}),
                'Test': test_results.get('basic_metrics', {})
            }
            
            visualizer.plot_metrics_comparison(
                metrics_comparison,
                save_path=charts_dir / "metrics_comparison.png"
            )
            plt.close('all')
        
        # Embedding visualization
        if hasattr(args, 'embedding_viz') and args.embedding_viz:
            visualizer.plot_embedding_visualization(
                train_embeddings[:1000],  # Limit for performance
                np.array(train_labels[:1000]),
                save_path=charts_dir / "embedding_visualization.png"
            )
            plt.close('all')
        
        # Interactive dashboard
        if hasattr(args, 'interactive_dashboard') and args.interactive_dashboard:
            dashboard_path = results_dir / "interactive" / "dashboard.html"
            visualizer.create_interactive_dashboard(
                test_results, save_path=dashboard_path
            )
        
        # Step 5: Save Models
        if config.save_models:
            logger.info("=" * 50)
            logger.info("STEP 5: Saving Models")
            logger.info("=" * 50)
            
            model_manager.save_models(results_dir / "models")
        
        # Step 6: Comprehensive Report
        logger.info("=" * 50)
        logger.info("STEP 6: Comprehensive Report")
        logger.info("=" * 50)
        
        experiment_info = {
            'name': config.experiment_name,
            'description': config.description,
            'datasets': config.data.datasets,
            'classifier': config.training.classifier_type,
            'model_name': config.model.model_name,
            'total_samples': len(train_examples) + len(val_examples) + len(test_examples)
        }
        
        # Create comprehensive HTML report
        visualizer.create_comprehensive_report(
            test_results, experiment_info, results_dir
        )
        
        # Save experiment summary
        visualizer.save_experiment_summary(
            experiment_results, experiment_info, results_dir
        )
        
        experiment_results['success'] = True
        experiment_results['end_time'] = time.time()
        experiment_results['duration'] = experiment_results['end_time'] - experiment_results['start_time']
        
        logger.info(f"Enhanced experiment completed successfully in {experiment_results['duration']:.2f} seconds")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"View comprehensive report: {results_dir / 'reports' / 'comprehensive_report.html'}")
        
    except Exception as e:
        logger.error(f"Enhanced experiment failed: {e}")
        experiment_results['success'] = False
        experiment_results['error'] = str(e)
        experiment_results['end_time'] = time.time()
        experiment_results['duration'] = experiment_results['end_time'] - experiment_results['start_time']
        raise
    
    return experiment_results


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Run enhanced experiment
        results = run_enhanced_experiment(config, args)
        
        if results.get('success', False):
            logger.info("🎉 Enhanced experiment completed successfully!")
            logger.info("📊 Beautiful visualizations and reports generated!")
            return 0
        else:
            logger.error("❌ Enhanced experiment failed!")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    sys.exit(main())
