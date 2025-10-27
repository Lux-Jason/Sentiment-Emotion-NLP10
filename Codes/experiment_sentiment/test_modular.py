#!/usr/bin/env python3
"""
Test script for the modular pseudo-civility detection implementation.

This script performs basic validation of the modular components
without requiring heavy model downloads or extensive computation.
"""
import logging
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Codes.experiment_sentiment.config import ExperimentConfig, get_default_config_binary
from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample, TextPreprocessor
from Codes.experiment_sentiment.evaluation import MetricsCalculator, ErrorAnalyzer, Evaluator


def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def test_config():
    """Test configuration management."""
    print("=" * 50)
    print("Testing Configuration Management")
    print("=" * 50)
    
    try:
        # Test default config creation
        config = get_default_config_binary()
        print(f"✅ Default config created: {config.experiment_name}")
        
        # Test config serialization
        config_dict = config.to_dict()
        print(f"✅ Config serialized to dict with {len(config_dict)} keys")
        
        # Test config loading from dict
        new_config = ExperimentConfig.from_dict(config_dict)
        print(f"✅ Config loaded from dict: {new_config.experiment_name}")
        
        # Test config file operations
        test_config_path = Path("test_config.json")
        config.save_to_file(test_config_path)
        print(f"✅ Config saved to file: {test_config_path}")
        
        loaded_config = ExperimentConfig.load_from_file(test_config_path)
        print(f"✅ Config loaded from file: {loaded_config.experiment_name}")
        
        # Cleanup
        if test_config_path.exists():
            test_config_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_data_processor():
    """Test data processing components."""
    print("=" * 50)
    print("Testing Data Processing Components")
    print("=" * 50)
    
    try:
        # Test text preprocessor
        preprocessor = TextPreprocessor()
        
        test_texts = [
            "This is a great comment!",
            "This is a terrible comment with http://spam.com",
            "Short",
            "This is a very long comment " * 100,  # Too long
            "user@example.com should be removed"
        ]
        
        processed_texts = [preprocessor.preprocess(text) for text in test_texts]
        print(f"✅ Text preprocessing completed for {len(test_texts)} texts")
        for i, (original, processed) in enumerate(zip(test_texts, processed_texts)):
            print(f"  {i+1}: '{original[:50]}...' -> '{processed[:50]}...'")
        
        # Test data example creation
        examples = [
            DataExample(text="Good comment", label=1, source="test"),
            DataExample(text="Bad comment", label=0, source="test"),
        ]
        print(f"✅ Created {len(examples)} DataExample objects")
        
        # Test data processor initialization
        config = get_default_config_binary()
        config.data.datasets = []  # No datasets for quick test
        processor = DataProcessor(config)
        print(f"✅ DataProcessor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False


def test_evaluation():
    """Test evaluation components."""
    print("=" * 50)
    print("Testing Evaluation Components")
    print("=" * 50)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()
        # Flip some predictions for realistic evaluation
        flip_indices = np.random.choice(n_samples, size=20, replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        y_proba = np.random.rand(n_samples, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
        
        print(f"✅ Created synthetic evaluation data: {n_samples} samples")
        print(f"  Accuracy: {np.mean(y_true == y_pred):.3f}")
        
        # Test metrics calculator
        metrics_calc = MetricsCalculator(average="binary")
        basic_metrics = metrics_calc.calculate_basic_metrics(y_true, y_pred)
        print(f"✅ Basic metrics calculated: {list(basic_metrics.keys())}")
        
        prob_metrics = metrics_calc.calculate_probabilistic_metrics(y_true, y_proba)
        print(f"✅ Probabilistic metrics calculated: {list(prob_metrics.keys())}")
        
        cal_metrics = metrics_calc.calculate_calibration_metrics(y_true, y_proba)
        print(f"✅ Calibration metrics calculated: {list(cal_metrics.keys())}")
        
        # Test error analyzer
        error_analyzer = ErrorAnalyzer()
        error_analysis = error_analyzer.analyze_errors(y_true, y_pred, y_proba)
        print(f"✅ Error analysis completed: {list(error_analysis.keys())}")
        
        # Test evaluator
        config = get_default_config_binary()
        evaluator = Evaluator(config)
        
        # Create test texts and metadata
        test_texts = [f"Test comment {i}" for i in range(n_samples)]
        test_metadata = [{"source": "test", "id": i} for i in range(n_samples)]
        
        results = evaluator.evaluate(
            y_true, y_pred, y_proba, test_texts, test_metadata
        )
        print(f"✅ Full evaluation completed: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False


def test_integration():
    """Test basic integration between components."""
    print("=" * 50)
    print("Testing Component Integration")
    print("=" * 50)
    
    try:
        # Create a minimal end-to-end test
        config = get_default_config_binary()
        config.data.datasets = []  # Skip real data loading
        config.data.max_samples = 10  # Very small for testing
        
        # Test data processor with synthetic data
        processor = DataProcessor(config)
        
        # Create synthetic examples
        synthetic_examples = [
            DataExample(text="This is a wonderful and helpful comment!", label=1, source="synthetic"),
            DataExample(text="This is terrible and rude.", label=0, source="synthetic"),
            DataExample(text="I disagree but respect your opinion.", label=1, source="synthetic"),
            DataExample(text="You are wrong and stupid!", label=0, source="synthetic"),
            DataExample(text="Great point, thanks for sharing.", label=1, source="synthetic"),
        ]
        
        # Test data splitting
        train_examples, val_examples, test_examples = processor.splitter.split_data(
            synthetic_examples, test_size=0.4, val_size=0.2
        )
        
        print(f"✅ Data splitting successful:")
        print(f"  Train: {len(train_examples)} examples")
        print(f"  Val: {len(val_examples)} examples")
        print(f"  Test: {len(test_examples)} examples")
        
        # Test data statistics
        stats = processor.get_data_statistics(train_examples, val_examples, test_examples)
        print(f"✅ Data statistics generated: {list(stats.keys())}")
        
        # Test evaluation pipeline
        if len(test_examples) > 0:
            # Create dummy predictions and probabilities
            test_labels = np.array([ex.label for ex in test_examples])
            dummy_predictions = np.random.randint(0, 2, len(test_examples))
            dummy_probabilities = np.random.rand(len(test_examples), 2)
            dummy_probabilities = dummy_probabilities / dummy_probabilities.sum(axis=1, keepdims=True)
            
            evaluator = Evaluator(config)
            eval_results = evaluator.evaluate(
                test_labels, dummy_predictions, dummy_probabilities
            )
            print(f"✅ Integration evaluation successful: {list(eval_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Starting Modular Implementation Tests")
    print("=" * 60)
    
    setup_logging()
    
    tests = [
        ("Configuration Management", test_config),
        ("Data Processing", test_data_processor),
        ("Evaluation", test_evaluation),
        ("Integration", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} tests...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} tests failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} test suites passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The modular implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
