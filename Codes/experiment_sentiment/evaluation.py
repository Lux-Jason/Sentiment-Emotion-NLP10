"""
Advanced evaluation module for pseudo-civility detection.
Includes comprehensive metrics, error analysis, and visualization.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import with fallbacks
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, roc_curve, auc
    )
    from sklearn.calibration import calibration_curve
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


class MetricsCalculator:
    """Advanced metrics calculation for classification tasks."""
    
    def __init__(self, average: str = "macro"):
        self.average = average
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for metrics calculation")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average=self.average, zero_division=0),
            'precision': precision_score(y_true, y_pred, average=self.average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average, zero_division=0)
        }
        
        # Add per-class metrics for multi-class
        if len(np.unique(y_true)) > 2:
            for class_label in np.unique(y_true):
                class_mask = (y_true == class_label)
                if class_mask.sum() > 0:
                    metrics[f'accuracy_class_{class_label}'] = accuracy_score(
                        y_true[class_mask], y_pred[class_mask]
                    )
                    metrics[f'f1_class_{class_label}'] = f1_score(
                        y_true[class_mask], y_pred[class_mask], zero_division=0
                    )
        
        return metrics
    
    def calculate_probabilistic_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics that require probabilities."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for metrics calculation")
        
        metrics = {}
        
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_proba.shape[1] == 2:
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba[:, 0]
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_positive)
                
                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
                metrics['pr_auc'] = auc(recall, precision)
            
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average=self.average
                )
        
        except Exception as e:
            logger.warning(f"Failed to calculate probabilistic metrics: {e}")
        
        return metrics
    
    def calculate_calibration_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_proba: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration metrics."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for calibration metrics")
        
        metrics = {}
        
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification calibration
                if y_proba.shape[1] == 2:
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba[:, 0]
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_proba_positive, n_bins=n_bins
                )
                
                # Expected Calibration Error (ECE)
                ece = self._calculate_ece(y_true, y_proba_positive, n_bins)
                metrics['expected_calibration_error'] = ece
                
                # Store calibration curve data for plotting
                metrics['calibration_curve'] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
        
        except Exception as e:
            logger.warning(f"Failed to calculate calibration metrics: {e}")
        
        return metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Calculate average confidence in this bin
                avg_confidence_in_bin = y_proba[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class ErrorAnalyzer:
    """Advanced error analysis for classification results."""
    
    def __init__(self):
        self.error_types = {}
    
    def analyze_errors(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None,
                      texts: Optional[List[str]] = None,
                      metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Perform comprehensive error analysis."""
        analysis = {
            'confusion_matrix': self._analyze_confusion_matrix(y_true, y_pred),
            'error_patterns': self._analyze_error_patterns(y_true, y_pred),
            'confidence_analysis': self._analyze_confidence(y_true, y_pred, y_proba),
            'sample_errors': self._analyze_sample_errors(y_true, y_pred, texts, metadata)
        }
        
        return analysis
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix patterns."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        n_classes = cm.shape[0]
        per_class_analysis = {}
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            per_class_analysis[f'class_{i}'] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
        
        return {
            'matrix': cm.tolist(),
            'per_class': per_class_analysis,
            'most_confused_pairs': self._find_most_confused_pairs(cm)
        }
    
    def _find_most_confused_pairs(self, cm: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find the most confused class pairs."""
        n_classes = cm.shape[0]
        confused_pairs = []
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    confused_pairs.append((i, j, int(cm[i, j])))
        
        # Sort by confusion count (descending)
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs[:5]  # Top 5 confused pairs
    
    def _analyze_error_patterns(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze systematic error patterns."""
        errors = y_true != y_pred
        correct = ~errors
        
        patterns = {
            'total_errors': int(errors.sum()),
            'total_correct': int(correct.sum()),
            'error_rate': float(errors.mean()),
            'accuracy': float(correct.mean())
        }
        
        # Per-class error analysis
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            class_mask = y_true == cls
            class_errors = errors[class_mask]
            class_correct = correct[class_mask]
            
            patterns[f'class_{cls}_errors'] = int(class_errors.sum())
            patterns[f'class_{cls}_correct'] = int(class_correct.sum())
            patterns[f'class_{cls}_error_rate'] = float(class_errors.mean() if class_mask.sum() > 0 else 0)
        
        return patterns
    
    def _analyze_confidence(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze confidence-related patterns."""
        if y_proba is None:
            return {}
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(y_proba, axis=1)
        
        # Analyze confidence for correct vs incorrect predictions
        errors = y_true != y_pred
        correct = ~errors
        
        analysis = {
            'avg_confidence_correct': float(confidence_scores[correct].mean()) if correct.sum() > 0 else 0,
            'avg_confidence_errors': float(confidence_scores[errors].mean()) if errors.sum() > 0 else 0,
            'avg_confidence_overall': float(confidence_scores.mean()),
            'confidence_std': float(confidence_scores.std())
        }
        
        # Confidence distribution analysis
        confidence_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
        confidence_distribution = {}
        
        for i in range(len(confidence_bins) - 1):
            lower, upper = confidence_bins[i], confidence_bins[i + 1]
            mask = (confidence_scores >= lower) & (confidence_scores < upper)
            
            if mask.sum() > 0:
                accuracy = y_true[mask] == y_pred[mask]
                confidence_distribution[f'{lower}-{upper}'] = {
                    'count': int(mask.sum()),
                    'accuracy': float(accuracy.mean())
                }
        
        analysis['confidence_distribution'] = confidence_distribution
        
        return analysis
    
    def _analyze_sample_errors(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             texts: Optional[List[str]] = None,
                             metadata: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze individual error samples."""
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        sample_analysis = {
            'total_error_samples': len(error_indices),
            'error_samples': []
        }
        
        # Analyze a subset of error samples
        max_samples = min(10, len(error_indices))
        selected_indices = error_indices[:max_samples]
        
        for idx in selected_indices:
            error_info = {
                'index': int(idx),
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx])
            }
            
            if texts:
                error_info['text'] = texts[idx]
            
            if metadata and idx < len(metadata):
                error_info['metadata'] = metadata[idx]
            
            sample_analysis['error_samples'].append(error_info)
        
        return sample_analysis


class Visualizer:
    """Advanced visualization for evaluation results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    def plot_confusion_matrix(self, 
                           cm: np.ndarray, 
                           class_names: Optional[List[str]] = None,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self, 
                            fraction_of_positives: np.ndarray,
                            mean_predicted_value: np.ndarray,
                            save_path: Optional[Path] = None) -> plt.Figure:
        """Plot calibration curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               label="Calibration curve", color=self.colors[0])
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, 
                      fpr: np.ndarray, 
                      tpr: np.ndarray, 
                      roc_auc: float,
                      save_path: Optional[Path] = None) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color=self.colors[0], lw=2,
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                  precision: np.ndarray, 
                                  recall: np.ndarray, 
                                  pr_auc: float,
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color=self.colors[0], lw=2,
               label=f'PR curve (AUC = {pr_auc:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, config):
        self.config = config
        self.metrics_calculator = MetricsCalculator(config.evaluation.average)
        self.error_analyzer = ErrorAnalyzer()
        self.visualizer = Visualizer()
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray, 
                y_proba: Optional[np.ndarray] = None,
                texts: Optional[List[str]] = None,
                metadata: Optional[List[Dict]] = None,
                save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")
        
        results = {
            'basic_metrics': {},
            'probabilistic_metrics': {},
            'calibration_metrics': {},
            'error_analysis': {},
            'classification_report': {}
        }
        
        # Basic metrics
        if 'accuracy' in self.config.evaluation.metrics or 'f1' in self.config.evaluation.metrics:
            results['basic_metrics'] = self.metrics_calculator.calculate_basic_metrics(y_true, y_pred)
        
        # Probabilistic metrics
        if y_proba is not None and 'roc_auc' in self.config.evaluation.metrics:
            results['probabilistic_metrics'] = self.metrics_calculator.calculate_probabilistic_metrics(y_true, y_proba)
        
        # Calibration metrics
        if (y_proba is not None and 
            self.config.evaluation.calibration and 
            'calibration' in self.config.evaluation.metrics):
            results['calibration_metrics'] = self.metrics_calculator.calculate_calibration_metrics(y_true, y_proba)
        
        # Error analysis
        if self.config.evaluation.error_analysis:
            results['error_analysis'] = self.error_analyzer.analyze_errors(y_true, y_pred, y_proba, texts, metadata)
        
        # Classification report
        if SKLEARN_AVAILABLE:
            results['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        
        # Generate visualizations
        if save_dir:
            self._generate_visualizations(results, y_true, y_pred, y_proba, save_dir)
        
        # Save results
        if save_dir and self.config.evaluation.save_results:
            self._save_results(results, save_dir)
        
        logger.info("Evaluation completed")
        return results
    
    def _generate_visualizations(self, 
                               results: Dict[str, Any],
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray],
                               save_dir: Path):
        """Generate and save visualizations."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        if 'confusion_matrix' in results.get('error_analysis', {}):
            cm_data = results['error_analysis']['confusion_matrix']
            cm = np.array(cm_data['matrix'])
            
            fig = self.visualizer.plot_confusion_matrix(
                cm, save_path=save_dir / 'confusion_matrix.png'
            )
            plt.close(fig)
        
        # Calibration curve
        if 'calibration_curve' in results.get('calibration_metrics', {}):
            cal_data = results['calibration_metrics']['calibration_curve']
            fraction_of_positives = np.array(cal_data['fraction_of_positives'])
            mean_predicted_value = np.array(cal_data['mean_predicted_value'])
            
            fig = self.visualizer.plot_calibration_curve(
                fraction_of_positives, mean_predicted_value,
                save_path=save_dir / 'calibration_curve.png'
            )
            plt.close(fig)
        
        # ROC curve
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                if y_proba.shape[1] == 2:
                    y_proba_positive = y_proba[:, 1]
                else:
                    y_proba_positive = y_proba[:, 0]
                
                fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
                roc_auc = results['probabilistic_metrics'].get('roc_auc', 0.0)
                
                fig = self.visualizer.plot_roc_curve(
                    fpr, tpr, roc_auc, save_path=save_dir / 'roc_curve.png'
                )
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate ROC curve: {e}")
    
    def _save_results(self, results: Dict[str, Any], save_dir: Path):
        """Save evaluation results to files."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        with open(save_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary report
        self._save_summary_report(results, save_dir)
        
        logger.info(f"Evaluation results saved to {save_dir}")
    
    def _save_summary_report(self, results: Dict[str, Any], save_dir: Path):
        """Save a human-readable summary report."""
        report_lines = ["# Evaluation Summary Report\n"]
        
        # Basic metrics
        if results['basic_metrics']:
            report_lines.append("## Basic Metrics\n")
            for metric, value in results['basic_metrics'].items():
                report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # Probabilistic metrics
        if results['probabilistic_metrics']:
            report_lines.append("## Probabilistic Metrics\n")
            for metric, value in results['probabilistic_metrics'].items():
                report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # Calibration metrics
        if results['calibration_metrics']:
            report_lines.append("## Calibration Metrics\n")
            for metric, value in results['calibration_metrics'].items():
                if metric != 'calibration_curve':
                    report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")
        
        # Error analysis summary
        if results['error_analysis']:
            error_patterns = results['error_analysis'].get('error_patterns', {})
            if error_patterns:
                report_lines.append("## Error Analysis\n")
                report_lines.append(f"- Total errors: {error_patterns.get('total_errors', 0)}")
                report_lines.append(f"- Error rate: {error_patterns.get('error_rate', 0):.4f}")
                report_lines.append("")
        
        # Write report
        with open(save_dir / 'evaluation_summary.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
