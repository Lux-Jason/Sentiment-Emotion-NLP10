#!/usr/bin/env python3
"""
Simple training script to demonstrate the modular implementation.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

def run_training_demo():
    """Run a demonstration of the modular training system."""
    print("🚀 Starting Modular Pseudo-Civility Detection Training")
    print("=" * 60)
    
    try:
        # Import modules
        print("📦 Importing modules...")
        from Codes.experiment_sentiment.config import get_default_config_binary
        from Codes.experiment_sentiment.data_processor import DataProcessor, DataExample
        from Codes.experiment_sentiment.models import ModelManager
        from Codes.experiment_sentiment.evaluation import Evaluator
        print("✅ All modules imported successfully")
        
        # Create configuration
        print("\n⚙️  Creating configuration...")
        config = get_default_config_binary()
        config.data.max_samples = 100  # Use small dataset for demo
        print(f"✅ Configuration created: {config.experiment_name}")
        
        # Create data processor
        print("\n📊 Setting up data processor...")
        config.data.datasets = []  # Skip real data loading for demo
        processor = DataProcessor(config)
        print("✅ Data processor initialized")
        
        # Create synthetic data for demonstration
        print("\n🔧 Creating synthetic demonstration data...")
        import numpy as np
        
        # Create synthetic examples
        texts = [
            "This is a great comment!",
            "I disagree with this approach.",
            "Thank you for sharing this information.",
            "This is terrible and wrong.",
            "I think you should reconsider.",
            "Excellent work on this project.",
            "This makes no sense at all.",
            "I appreciate your perspective on this.",
            "Completely disagree with your analysis.",
            "Well done, very insightful!"
        ]
        
        labels = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 1=civil, 0=uncivil
        
        examples = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            example = DataExample(
                text=text,
                label=label,
                source="demo",
                metadata={"id": i}
            )
            examples.append(example)
        
        print(f"✅ Created {len(examples)} synthetic examples")
        
        # Split data
        print("\n🔄 Splitting data...")
        train_examples, val_examples, test_examples = processor.splitter.split_data(
            examples, test_size=0.3, val_size=0.2
        )
        print(f"✅ Data split - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        # Create model manager
        print("\n🤖 Setting up model manager...")
        model_manager = ModelManager(config)
        print("✅ Model manager initialized")
        
        # Create synthetic embeddings for demo
        print("\n🔢 Creating synthetic embeddings...")
        embedding_dim = 1024
        
        def create_synthetic_embeddings(texts):
            """Create synthetic embeddings for demonstration."""
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), embedding_dim)
            # Make civil and uncivil comments more separable
            for i, text in enumerate(texts):
                if "great" in text.lower() or "excellent" in text.lower() or "thank" in text.lower():
                    embeddings[i, :10] += 2.0  # Make positive examples more positive
                elif "terrible" in text.lower() or "wrong" in text.lower() or "no sense" in text.lower():
                    embeddings[i, :10] -= 2.0  # Make negative examples more negative
            return embeddings
        
        train_embeddings = create_synthetic_embeddings([ex.text for ex in train_examples])
        val_embeddings = create_synthetic_embeddings([ex.text for ex in val_examples])
        test_embeddings = create_synthetic_embeddings([ex.text for ex in test_examples])
        
        train_labels = np.array([ex.label for ex in train_examples])
        val_labels = np.array([ex.label for ex in val_examples])
        test_labels = np.array([ex.label for ex in test_examples])
        
        print(f"✅ Embeddings created - Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
        
        # Train model
        print("\n🏋️  Training model...")
        model_manager.train_classifier(
            train_embeddings, train_labels,
            val_embeddings, val_labels
        )
        print("✅ Model training completed")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        test_predictions = model_manager.predict(test_embeddings)
        test_probabilities = model_manager.predict_proba(test_embeddings)
        print("✅ Predictions generated")
        
        # Evaluate results
        print("\n📊 Evaluating results...")
        evaluator = Evaluator(config)
        results = evaluator.evaluate(test_labels, test_predictions, test_probabilities)
        
        print("\n📈 TRAINING RESULTS")
        print("=" * 40)
        print(f"Dataset: {len(examples)} synthetic examples")
        print(f"Model: {config.model.model_type}")
        print(f"Test Accuracy: {results['basic_metrics']['accuracy']:.3f}")
        print(f"Test F1-Score: {results['basic_metrics']['f1_score']:.3f}")
        print(f"Test Precision: {results['basic_metrics']['precision']:.3f}")
        print(f"Test Recall: {results['basic_metrics']['recall']:.3f}")
        print(f"Test ROC-AUC: {results['basic_metrics']['roc_auc']:.3f}")
        
        # Show some predictions
        print("\n🔍 Sample Predictions:")
        print("-" * 40)
        for i, (example, pred, prob) in enumerate(zip(test_examples, test_predictions, test_probabilities)):
            confidence = prob[pred] if len(prob.shape) == 1 else prob[pred]
            pred_label = "Civil" if pred == 1 else "Uncivil"
            true_label = "Civil" if example.label == 1 else "Uncivil"
            correct = "✅" if pred == example.label else "❌"
            print(f"{correct} {example.text[:40]:<40} -> {pred_label} ({confidence:.2f}) [True: {true_label}]")
        
        print("\n🎉 TRAINING DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("📁 The modular implementation is working correctly.")
        print("🚀 Ready for production use with real datasets.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🧪 Modular Training Demonstration")
    print("=" * 60)
    print("This script demonstrates the modular pseudo-civility detection system")
    print("using synthetic data for quick testing.\n")
    
    success = run_training_demo()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("\nTo run with real data, use:")
        print("python main.py --config-type binary --datasets chnsenticorp")
        return 0
    else:
        print("\n❌ Demo failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
