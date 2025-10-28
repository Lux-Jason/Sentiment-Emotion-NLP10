#!/usr/bin/env python3
"""
Demo script to showcase enhanced visualization system.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Codes.experiment_sentiment.enhanced_visualizer import EnhancedVisualizer


def create_demo_data():
    """Create demo data for visualization testing."""
    np.random.seed(42)
    
    # Create sample confusion matrix
    cm = np.array([[85, 15], [10, 90]])
    
    # Create sample ROC curve data
    fpr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr = np.array([0.0, 0.6, 0.75, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0])
    roc_auc = 0.85
    
    # Create sample Precision-Recall curve data
    precision = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])
    recall = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    pr_auc = 0.78
    
    # Create sample metrics
    metrics_comparison = {
        'Model A': {'accuracy': 0.85, 'f1': 0.82, 'precision': 0.84, 'recall': 0.80},
        'Model B': {'accuracy': 0.88, 'f1': 0.86, 'precision': 0.87, 'recall': 0.85},
        'Model C': {'accuracy': 0.82, 'f1': 0.79, 'precision': 0.81, 'recall': 0.77}
    }
    
    # Create sample embeddings
    embeddings = np.random.randn(200, 50)
    labels = np.random.randint(0, 3, 200)
    
    # Create sample results
    results = {
        'basic_metrics': {
            'accuracy': 0.875,
            'f1': 0.862,
            'precision': 0.874,
            'recall': 0.850
        },
        'probabilistic_metrics': {
            'roc_auc': 0.85,
            'pr_auc': 0.78
        },
        'error_analysis': {
            'error_patterns': {
                'total_errors': 25,
                'error_rate': 0.125,
                'accuracy': 0.875
            },
            'confusion_matrix': {
                'matrix': cm.tolist()
            }
        }
    }
    
    return {
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
        'metrics_comparison': metrics_comparison,
        'embeddings': embeddings,
        'labels': labels,
        'results': results
    }


def main():
    """Run demo of enhanced visualization system."""
    print("🎨 Enhanced Visualization System Demo")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = EnhancedVisualizer()
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Create results directory
    results_dir = visualizer.create_results_directory(
        Path("outputs"), 
        "enhanced_viz_demo"
    )
    
    charts_dir = results_dir / "charts"
    
    print(f"📁 Results will be saved to: {results_dir}")
    
    # 1. Enhanced Confusion Matrix
    print("📊 Creating enhanced confusion matrix...")
    fig = visualizer.plot_enhanced_confusion_matrix(
        demo_data['cm'],
        class_names=['Negative', 'Positive'],
        title="Enhanced Confusion Matrix Demo",
        save_path=charts_dir / "enhanced_confusion_matrix.png"
    )
    plt.close(fig)
    print("✅ Enhanced confusion matrix saved!")
    
    # 2. Enhanced ROC Curve
    print("📈 Creating enhanced ROC curve...")
    fig = visualizer.plot_enhanced_roc_curve(
        demo_data['fpr'],
        demo_data['tpr'],
        demo_data['roc_auc'],
        title="Enhanced ROC Curve Demo",
        save_path=charts_dir / "enhanced_roc_curve.png"
    )
    plt.close(fig)
    print("✅ Enhanced ROC curve saved!")
    
    # 3. Enhanced Precision-Recall Curve
    print("📉 Creating enhanced Precision-Recall curve...")
    fig = visualizer.plot_enhanced_precision_recall_curve(
        demo_data['precision'],
        demo_data['recall'],
        demo_data['pr_auc'],
        title="Enhanced Precision-Recall Curve Demo",
        save_path=charts_dir / "enhanced_pr_curve.png"
    )
    plt.close(fig)
    print("✅ Enhanced Precision-Recall curve saved!")
    
    # 4. Metrics Comparison
    print("📊 Creating metrics comparison chart...")
    fig = visualizer.plot_metrics_comparison(
        demo_data['metrics_comparison'],
        title="Model Performance Comparison Demo",
        save_path=charts_dir / "metrics_comparison.png"
    )
    plt.close(fig)
    print("✅ Metrics comparison chart saved!")
    
    # 5. Embedding Visualization
    print("🔍 Creating embedding visualization...")
    fig = visualizer.plot_embedding_visualization(
        demo_data['embeddings'],
        demo_data['labels'],
        method="tsne",
        title="t-SNE Embedding Visualization Demo",
        save_path=charts_dir / "embedding_visualization.png"
    )
    plt.close(fig)
    print("✅ Embedding visualization saved!")
    
    # 6. Interactive Dashboard
    print("🌐 Creating interactive dashboard...")
    dashboard_path = results_dir / "interactive" / "dashboard.html"
    visualizer.create_interactive_dashboard(
        demo_data['results'], save_path=dashboard_path
    )
    print("✅ Interactive dashboard saved!")
    
    # 7. Comprehensive HTML Report
    print("📄 Creating comprehensive HTML report...")
    experiment_info = {
        'name': 'Enhanced Visualization Demo',
        'description': 'Demo showcasing enhanced visualization capabilities',
        'datasets': ['Demo Dataset'],
        'classifier': 'Demo Classifier',
        'model_name': 'Demo Model',
        'total_samples': 1000
    }
    
    report_path = visualizer.create_comprehensive_report(
        demo_data['results'], experiment_info, results_dir
    )
    print("✅ Comprehensive HTML report saved!")
    
    # 8. Experiment Summary
    print("💾 Saving experiment summary...")
    visualizer.save_experiment_summary(
        demo_data['results'], experiment_info, results_dir
    )
    print("✅ Experiment summary saved!")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)
    print(f"📁 Results directory: {results_dir}")
    print(f"📊 Charts: {charts_dir}")
    print(f"🌐 Interactive dashboard: {dashboard_path}")
    print(f"📄 HTML report: {report_path}")
    print("\n🔍 View the generated files to see the enhanced visualizations!")


if __name__ == "__main__":
    main()
