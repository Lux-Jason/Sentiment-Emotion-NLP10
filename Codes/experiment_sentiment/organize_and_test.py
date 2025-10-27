#!/usr/bin/env python3
"""
Script to organize old files and test the new modular implementation.
"""
import os
import shutil
import sys
from pathlib import Path

def organize_files():
    """Move old files to draft folder."""
    print("🗂️  Organizing files...")
    
    # Create draft directory
    draft_dir = Path("draft")
    draft_dir.mkdir(exist_ok=True)
    
    # Files to move
    files_to_move = [
        "train_sentiment.py",
        "prepare_datasets.py"
    ]
    
    moved_files = []
    for file_name in files_to_move:
        source_path = Path(file_name)
        if source_path.exists():
            dest_path = draft_dir / file_name
            shutil.move(str(source_path), str(dest_path))
            moved_files.append(file_name)
            print(f"  ✅ Moved {file_name} to draft/")
        else:
            print(f"  ⚠️  {file_name} not found")
    
    print(f"📁 File organization completed. Moved {len(moved_files)} files.")
    return len(moved_files) > 0

def test_basic_imports():
    """Test basic imports of the modular implementation."""
    print("\n🧪 Testing basic imports...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parents[2]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        
        # Test imports
        from Codes.experiment_sentiment.config import ExperimentConfig, get_default_config_binary
        print("  ✅ Config module imported successfully")
        
        from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample
        print("  ✅ Data processor module imported successfully")
        
        from Codes.experiment_sentiment.models import ModelManager
        print("  ✅ Models module imported successfully")
        
        from Codes.experiment_sentiment.evaluation import Evaluator, MetricsCalculator
        print("  ✅ Evaluation module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation and validation."""
    print("\n⚙️  Testing configuration management...")
    
    try:
        from Codes.experiment_sentiment.config import get_default_config_binary
        
        # Create default config
        config = get_default_config_binary()
        print(f"  ✅ Default config created: {config.experiment_name}")
        
        # Test config attributes
        assert hasattr(config, 'data'), "Config missing data attribute"
        assert hasattr(config, 'model'), "Config missing model attribute"
        assert hasattr(config, 'training'), "Config missing training attribute"
        assert hasattr(config, 'evaluation'), "Config missing evaluation attribute"
        print("  ✅ Config structure validated")
        
        # Test config serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "Config serialization failed"
        print("  ✅ Config serialization works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_data_processing():
    """Test data processing components."""
    print("\n📊 Testing data processing...")
    
    try:
        from Codes.experiment_sentiment.config import get_default_config_binary
        from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample, TextPreprocessor
        
        # Test text preprocessor
        preprocessor = TextPreprocessor()
        test_text = "This is a TEST comment with http://example.com and user@test.com!"
        processed = preprocessor.preprocess(test_text)
        print(f"  ✅ Text preprocessing: '{test_text[:30]}...' -> '{processed[:30]}...'")
        
        # Test data example creation
        example = DataExample(text="Test comment", label=1, source="test")
        print(f"  ✅ DataExample created: {example}")
        
        # Test data processor initialization
        config = get_default_config_binary()
        config.data.datasets = []  # Skip real data loading
        processor = DataProcessor(config)
        print("  ✅ DataProcessor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing test failed: {e}")
        return False

def test_evaluation_components():
    """Test evaluation components."""
    print("\n📈 Testing evaluation components...")
    
    try:
        import numpy as np
        from Codes.experiment_sentiment.config import get_default_config_binary
        from Codes.experiment_sentiment.evaluation import MetricsCalculator, ErrorAnalyzer, Evaluator
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 50
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true.copy()
        # Flip some predictions
        flip_indices = np.random.choice(n_samples, size=10, replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        y_proba = np.random.rand(n_samples, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        print(f"  ✅ Synthetic data created: {n_samples} samples")
        
        # Test metrics calculator
        metrics_calc = MetricsCalculator(average="binary")
        basic_metrics = metrics_calc.calculate_basic_metrics(y_true, y_pred)
        print(f"  ✅ Basic metrics calculated: {list(basic_metrics.keys())}")
        
        # Test error analyzer
        error_analyzer = ErrorAnalyzer()
        error_analysis = error_analyzer.analyze_errors(y_true, y_pred, y_proba)
        print(f"  ✅ Error analysis completed: {list(error_analysis.keys())}")
        
        # Test evaluator
        config = get_default_config_binary()
        evaluator = Evaluator(config)
        results = evaluator.evaluate(y_true, y_pred, y_proba)
        print(f"  ✅ Full evaluation completed: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        return False

def test_integration():
    """Test basic integration between components."""
    print("\n🔗 Testing component integration...")
    
    try:
        import numpy as np
        from Codes.experiment_sentiment.config import get_default_config_binary
        from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample
        from Codes.experiment_sentiment.evaluation import Evaluator
        
        # Create minimal test setup
        config = get_default_config_binary()
        config.data.datasets = []  # Skip real data loading
        config.data.max_samples = 10
        
        # Test data processor
        processor = DataProcessor(config)
        
        # Create synthetic examples
        examples = [
            DataExample(text="Great comment!", label=1, source="test"),
            DataExample(text="Terrible comment!", label=0, source="test"),
            DataExample(text="Neutral comment.", label=1, source="test"),
            DataExample(text="Rude comment!", label=0, source="test"),
        ]
        
        # Test data splitting
        train_examples, val_examples, test_examples = processor.splitter.split_data(
            examples, test_size=0.5, val_size=0.0
        )
        
        print(f"  ✅ Data splitting: Train={len(train_examples)}, Test={len(test_examples)}")
        
        # Test evaluation integration
        if len(test_examples) > 0:
            test_labels = np.array([ex.label for ex in test_examples])
            dummy_predictions = np.random.randint(0, 2, len(test_examples))
            dummy_probabilities = np.random.rand(len(test_examples), 2)
            dummy_probabilities = dummy_probabilities / dummy_probabilities.sum(axis=1, keepdims=True)
            
            evaluator = Evaluator(config)
            eval_results = evaluator.evaluate(test_labels, dummy_predictions, dummy_probabilities)
            print(f"  ✅ Integration evaluation: {list(eval_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all organization and testing tasks."""
    print("🚀 Starting File Organization and Testing")
    print("=" * 60)
    
    # Step 1: Organize files
    organize_success = organize_files()
    
    # Step 2: Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Management", test_config_creation),
        ("Data Processing", test_data_processing),
        ("Evaluation Components", test_evaluation_components),
        ("Integration", test_integration),
    ]
    
    print("\n🧪 Running Tests")
    print("=" * 60)
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
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
        print("📁 Files have been organized and the new system is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
