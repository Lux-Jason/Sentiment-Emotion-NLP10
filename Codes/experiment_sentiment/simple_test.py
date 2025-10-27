#!/usr/bin/env python3
"""
Simple test for the modular implementation.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from Codes.experiment_sentiment.config import ExperimentConfig, get_default_config_binary
        print("✅ Config imported")
        
        from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample
        print("✅ Data processor imported")
        
        from Codes.experiment_sentiment.models import ModelManager
        print("✅ Models imported")
        
        from Codes.experiment_sentiment.evaluation import Evaluator
        print("✅ Evaluation imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    
    try:
        from Codes.experiment_sentiment.config import get_default_config_binary
        
        config = get_default_config_binary()
        print(f"✅ Config created: {config.experiment_name}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_example():
    """Test data example creation."""
    print("\nTesting data example...")
    
    try:
        from Codes.experiment_sentiment.data_processor import DataExample
        
        example = DataExample(text="Test comment", label=1, source="test")
        print(f"✅ DataExample created: {example}")
        
        return True
    except Exception as e:
        print(f"❌ DataExample test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🧪 Simple Modular Implementation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_example,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Basic tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
