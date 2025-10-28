#!/usr/bin/env python3
"""
Enhanced visualization system with beautiful charts and automatic versioning.
Creates rich, publication-quality visualizations with modern design.
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import with fallbacks
try:
    from sklearn.metrics import (
        confusion_matrix, roc_curve, precision_recall_curve, auc
    )
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available. Install with: pip install pandas")

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """Enhanced visualization system with beautiful charts and modern design."""
    
    def __init__(self, theme: str = "modern"):
        self.theme = theme
        self.setup_style()
        self.color_palette = self._get_color_palette()
        
    def setup_style(self):
        """Setup modern matplotlib style."""
        # Set modern style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        # Set font properties
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.2,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.transparent': False,
            'savefig.facecolor': 'white'
        })
    
    def _get_color_palette(self) -> Dict[str, str]:
        """Get modern color palette."""
        return {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'danger': '#C73E1D',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'info': '#2196F3',
            'light': '#F5F5F5',
            'dark': '#263238',
            'gradient1': ['#667eea', '#764ba2'],
            'gradient2': ['#f093fb', '#f5576c'],
            'gradient3': ['#4facfe', '#00f2fe'],
            'gradient4': ['#43e97b', '#38f9d7']
        }
    
    def create_results_directory(self, base_dir: Path, experiment_name: str) -> Path:
        """Create a new results directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / "results" / f"{experiment_name}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (results_dir / "charts").mkdir(exist_ok=True)
        (results_dir / "interactive").mkdir(exist_ok=True)
        (results_dir / "reports").mkdir(exist_ok=True)
        
        logger.info(f"Created results directory: {results_dir}")
        return results_dir
    
    def plot_enhanced_confusion_matrix(self, 
                                     cm: np.ndarray, 
                                     class_names: Optional[List[str]] = None,
                                     title: str = "Confusion Matrix",
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """Create enhanced confusion matrix with beautiful styling."""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap with custom styling
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax,
                   cbar_kws={'shrink': 0.8, 'aspect': 30},
                   annot_kws={'size': 14, 'weight': 'bold'},
                   linewidths=0.5,
                   linecolor='white')
        
        # Add percentage annotations
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm_normalized[i, j] * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=10, color='gray')
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Rotate labels
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_enhanced_roc_curve(self, 
                              fpr: np.ndarray, 
                              tpr: np.ndarray, 
                              roc_auc: float,
                              title: str = "ROC Curve",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """Create enhanced ROC curve with beautiful styling."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve with gradient fill
        ax.plot(fpr, tpr, 
               color=self.color_palette['primary'], 
               linewidth=3,
               label=f'ROC Curve (AUC = {roc_auc:.3f})',
               alpha=0.8)
        
        # Add gradient fill under curve
        ax.fill_between(fpr, tpr, alpha=0.3, color=self.color_palette['primary'])
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 
               color=self.color_palette['dark'], 
               linewidth=2, 
               linestyle='--', 
               alpha=0.7,
               label='Random Classifier')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_enhanced_precision_recall_curve(self, 
                                           precision: np.ndarray, 
                                           recall: np.ndarray, 
                                           pr_auc: float,
                                           title: str = "Precision-Recall Curve",
                                           save_path: Optional[Path] = None) -> plt.Figure:
        """Create enhanced Precision-Recall curve with beautiful styling."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve with gradient fill
        ax.plot(recall, precision, 
               color=self.color_palette['secondary'], 
               linewidth=3,
               label=f'PR Curve (AUC = {pr_auc:.3f})',
               alpha=0.8)
        
        # Add gradient fill under curve
        ax.fill_between(recall, precision, alpha=0.3, color=self.color_palette['secondary'])
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc="lower left", fontsize=12, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_metrics_comparison(self, 
                               metrics_dict: Dict[str, Dict[str, float]],
                               title: str = "Model Performance Comparison",
                               save_path: Optional[Path] = None) -> plt.Figure:
        """Create beautiful metrics comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        models = list(metrics_dict.keys())
        metrics = list(metrics_dict[models[0]].keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        # Create bars for each model
        for i, model in enumerate(models):
            values = [metrics_dict[model][metric] for metric in metrics]
            bars = ax.bar(x + i * width, values, width, 
                         label=model, alpha=0.8,
                         color=self.color_palette[list(self.color_palette.keys())[i % len(self.color_palette)]])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Styling
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def plot_embedding_visualization(self, 
                                   embeddings: np.ndarray,
                                   labels: np.ndarray,
                                   method: str = "tsne",
                                   title: str = "Embedding Visualization",
                                   save_path: Optional[Path] = None) -> plt.Figure:
        """Create beautiful embedding visualization."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for embedding visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Reduce dimensionality
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Get unique labels and colors
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
        
        # Styling
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   results: Dict[str, Any],
                                   save_path: Optional[Path] = None) -> str:
        """Create interactive dashboard with Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Metrics Comparison', 'Class Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Confusion Matrix
        if 'error_analysis' in results and 'confusion_matrix' in results['error_analysis']:
            cm_data = results['error_analysis']['confusion_matrix']['matrix']
            fig.add_trace(
                go.Heatmap(z=cm_data, colorscale='Blues', name='Confusion Matrix'),
                row=1, col=1
            )
        
        # ROC Curve (placeholder - would need actual data)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'),
            row=1, col=2
        )
        
        # Metrics Comparison
        if 'basic_metrics' in results:
            metrics = list(results['basic_metrics'].keys())
            values = list(results['basic_metrics'].values())
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Metrics'),
                row=2, col=1
            )
        
        # Class Distribution (placeholder)
        fig.add_trace(
            go.Bar(x=['Class 0', 'Class 1'], y=[50, 50], name='Distribution'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive HTML
        if save_path:
            pyo.plot(fig, filename=str(save_path), auto_open=False)
            return str(save_path)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_comprehensive_report(self, 
                                  results: Dict[str, Any],
                                  experiment_info: Dict[str, Any],
                                  save_dir: Path) -> str:
        """Create comprehensive HTML report with beautiful design."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {experiment_info.get('name', 'Unknown')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid {self.color_palette['primary']};
        }}
        .header h1 {{
            color: {self.color_palette['primary']};
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            font-size: 1.2em;
            margin: 10px 0;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
            border-left: 4px solid {self.color_palette['primary']};
        }}
        .section h2 {{
            color: {self.color_palette['primary']};
            margin-top: 0;
            font-size: 1.8em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 3px solid {self.color_palette['primary']};
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: {self.color_palette['primary']};
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        .success {{ color: {self.color_palette['success']}; }}
        .warning {{ color: {self.color_palette['warning']}; }}
        .danger {{ color: {self.color_palette['danger']}; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Experiment Report</h1>
            <p><strong>{experiment_info.get('name', 'Unknown Experiment')}</strong></p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <div class="metrics-grid">
"""
        
        # Add metrics cards
        if 'basic_metrics' in results:
            for metric, value in results['basic_metrics'].items():
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                </div>
                """
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Visualizations</h2>
            <div class="chart-container">
                <h3>Confusion Matrix</h3>
                <img src="charts/confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="chart-container">
                <h3>ROC Curve</h3>
                <img src="charts/roc_curve.png" alt="ROC Curve">
            </div>
            <div class="chart-container">
                <h3>Precision-Recall Curve</h3>
                <img src="charts/precision_recall_curve.png" alt="Precision-Recall Curve">
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Error Analysis</h2>
"""
        
        # Add error analysis
        if 'error_analysis' in results and 'error_patterns' in results['error_analysis']:
            error_patterns = results['error_analysis']['error_patterns']
            html_content += f"""
            <p><strong>Total Errors:</strong> {error_patterns.get('total_errors', 0)}</p>
            <p><strong>Error Rate:</strong> {error_patterns.get('error_rate', 0):.4f}</p>
            <p><strong>Accuracy:</strong> {error_patterns.get('accuracy', 0):.4f}</p>
"""
        
        html_content += """
        </div>
        
        <div class="footer">
            <p>Report generated by Enhanced Visualization System</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = save_dir / "reports" / "comprehensive_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive HTML report saved to {report_path}")
        return str(report_path)
    
    def save_experiment_summary(self, 
                              results: Dict[str, Any],
                              experiment_info: Dict[str, Any],
                              save_dir: Path):
        """Save experiment summary as JSON."""
        summary = {
            'experiment_info': experiment_info,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'visualization_theme': self.theme
        }
        
        summary_path = save_dir / "experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment summary saved to {summary_path}")
