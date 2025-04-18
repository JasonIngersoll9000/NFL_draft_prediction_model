#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for quarterback model results.

This script creates various visualizations of the quarterback model results,
including model performance metrics, predicted vs. actual values,
feature importances, and top QB prospects.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Define directories
RESULTS_DIR = os.path.join(project_root, 'results', 'quarterback')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')
FEATURES_DIR = os.path.join(RESULTS_DIR, 'features')
MODELS_DIR = os.path.join(project_root, 'models', 'quarterback')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

def load_data():
    """Load all necessary data for visualizations."""
    data = {}
    
    # Load predictions
    predictions_path = os.path.join(PREDICTIONS_DIR, 'qb_predictions.csv')
    if os.path.exists(predictions_path):
        data['predictions'] = pd.read_csv(predictions_path)
        print(f"Loaded predictions for {len(data['predictions'])} quarterbacks")
    else:
        print(f"Warning: Predictions file not found at {predictions_path}")
        data['predictions'] = None
    
    # Load evaluation results
    eval_path = os.path.join(MODELS_DIR, 'latest', 'latest', 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            data['evaluation'] = json.load(f)
        print("Loaded model evaluation results")
    else:
        print(f"Warning: Evaluation results not found at {eval_path}")
        data['evaluation'] = None
    
    # Load feature importances
    feature_files = {
        'classification_rf': os.path.join(FEATURES_DIR, 'cv_classification_rf_importances.csv'),
        'classification_xgb': os.path.join(FEATURES_DIR, 'cv_classification_xgb_importances.csv'),
        'regression_rf': os.path.join(FEATURES_DIR, 'cv_regression_rf_importances.csv'),
        'regression_xgb': os.path.join(FEATURES_DIR, 'cv_regression_xgb_importances.csv'),
        'efficiency_rf': os.path.join(FEATURES_DIR, 'cv_efficiency_rf_importances.csv'),
        'efficiency_xgb': os.path.join(FEATURES_DIR, 'cv_efficiency_xgb_importances.csv')
    }
    
    data['feature_importances'] = {}
    for name, path in feature_files.items():
        if os.path.exists(path):
            data['feature_importances'][name] = pd.read_csv(path)
            print(f"Loaded {name} feature importances")
        else:
            print(f"Warning: Feature importances not found at {path}")
    
    return data

def visualize_model_metrics(evaluation_data):
    """Visualize model evaluation metrics."""
    if not evaluation_data:
        print("No evaluation data available for visualization")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot classification metrics
    if 'classification' in evaluation_data:
        class_metrics = []
        for model, metrics in evaluation_data['classification'].items():
            for metric, value in metrics.items():
                if metric.endswith('_mean'):
                    metric_name = metric.replace('_mean', '')
                    class_metrics.append({
                        'Model': model.upper(),
                        'Metric': metric_name.capitalize(),
                        'Value': value
                    })
        
        if class_metrics:
            df_class = pd.DataFrame(class_metrics)
            sns.barplot(x='Metric', y='Value', hue='Model', data=df_class, ax=axes[0])
            axes[0].set_title('Classification Metrics')
            axes[0].set_ylim(0, 1)
    
    # Plot regression metrics
    if 'regression' in evaluation_data:
        reg_metrics = []
        for model, metrics in evaluation_data['regression'].items():
            for metric, value in metrics.items():
                if metric.endswith('_mean'):
                    metric_name = metric.replace('_mean', '')
                    if metric_name == 'r2':
                        metric_name = 'R²'
                    reg_metrics.append({
                        'Model': model.upper(),
                        'Metric': metric_name.upper(),
                        'Value': value
                    })
        
        if reg_metrics:
            df_reg = pd.DataFrame(reg_metrics)
            sns.barplot(x='Metric', y='Value', hue='Model', data=df_reg, ax=axes[1])
            axes[1].set_title('wAV Regression Metrics')
    
    # Plot efficiency metrics
    if 'efficiency' in evaluation_data:
        eff_metrics = []
        for model, metrics in evaluation_data['efficiency'].items():
            for metric, value in metrics.items():
                if metric.endswith('_mean'):
                    metric_name = metric.replace('_mean', '')
                    if metric_name == 'r2':
                        metric_name = 'R²'
                    eff_metrics.append({
                        'Model': model.upper(),
                        'Metric': metric_name.upper(),
                        'Value': value
                    })
        
        if eff_metrics:
            df_eff = pd.DataFrame(eff_metrics)
            sns.barplot(x='Metric', y='Value', hue='Model', data=df_eff, ax=axes[2])
            axes[2].set_title('wAV/Game Efficiency Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'model_metrics.png'), dpi=300)
    plt.close()
    print("Saved model metrics visualization")

def visualize_predictions(predictions_df):
    """Visualize predictions vs. actual values."""
    if predictions_df is None:
        print("No prediction data available for visualization")
        return
    
    # Filter to only include players with NFL experience
    nfl_players = predictions_df[predictions_df['actual_wav'] > 0].copy()
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Predicted vs. Actual wAV
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(nfl_players['actual_wav'], nfl_players['pred_wav'], alpha=0.7)
    
    # Add regression line
    x = nfl_players['actual_wav']
    y = nfl_players['pred_wav']
    m, b = np.polyfit(x, y, 1)
    ax1.plot(x, m*x + b, color='red', linestyle='--')
    
    # Calculate correlation
    corr = np.corrcoef(nfl_players['actual_wav'], nfl_players['pred_wav'])[0, 1]
    
    ax1.set_xlabel('Actual wAV')
    ax1.set_ylabel('Predicted wAV')
    ax1.set_title(f'Predicted vs. Actual wAV (Correlation: {corr:.3f})')
    
    # Add diagonal line (perfect prediction)
    max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.plot([0, max_val], [0, max_val], color='green', linestyle=':')
    
    # 2. Predicted vs. Actual wAV per game
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(nfl_players['actual_wav_per_game'], nfl_players['pred_wav_per_game'], alpha=0.7)
    
    # Add regression line
    x = nfl_players['actual_wav_per_game']
    y = nfl_players['pred_wav_per_game']
    m, b = np.polyfit(x, y, 1)
    ax2.plot(x, m*x + b, color='red', linestyle='--')
    
    # Calculate correlation
    corr = np.corrcoef(nfl_players['actual_wav_per_game'], nfl_players['pred_wav_per_game'])[0, 1]
    
    ax2.set_xlabel('Actual wAV/Game')
    ax2.set_ylabel('Predicted wAV/Game')
    ax2.set_title(f'Predicted vs. Actual wAV/Game (Correlation: {corr:.3f})')
    
    # Add diagonal line (perfect prediction)
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([0, max_val], [0, max_val], color='green', linestyle=':')
    
    # 3. Top 20 QBs by predicted wAV
    ax3 = fig.add_subplot(gs[1, :])
    
    # Sort by predicted wAV
    top_qbs = predictions_df.sort_values('pred_wav', ascending=False).head(20)
    
    # Create bar chart
    bar_width = 0.35
    x = np.arange(len(top_qbs))
    
    # Plot bars
    ax3.bar(x - bar_width/2, top_qbs['pred_wav'], bar_width, label='Predicted wAV')
    ax3.bar(x + bar_width/2, top_qbs['actual_wav'], bar_width, label='Actual wAV')
    
    # Add labels
    ax3.set_xlabel('Quarterback')
    ax3.set_ylabel('wAV')
    ax3.set_title('Top 20 QBs by Predicted wAV')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_qbs['name'], rotation=45, ha='right')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'predictions_visualization.png'), dpi=300)
    plt.close()
    print("Saved predictions visualization")

def visualize_feature_importances(feature_importances):
    """Visualize feature importances for different models."""
    if not feature_importances:
        print("No feature importance data available for visualization")
        return
    
    # Create visualizations for each model type
    model_types = ['classification', 'regression', 'efficiency']
    
    for model_type in model_types:
        # Check if we have data for this model type
        rf_key = f'{model_type}_rf'
        xgb_key = f'{model_type}_xgb'
        
        if rf_key not in feature_importances or xgb_key not in feature_importances:
            print(f"Missing feature importance data for {model_type}")
            continue
        
        # Get top 15 features for each model
        rf_features = feature_importances[rf_key].sort_values('importance', ascending=False).head(15)
        xgb_features = feature_importances[xgb_key].sort_values('importance', ascending=False).head(15)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot RF feature importances
        sns.barplot(x='importance', y='feature', data=rf_features, ax=axes[0])
        axes[0].set_title(f'Random Forest - Top 15 Features ({model_type.capitalize()})')
        axes[0].set_xlabel('Importance')
        axes[0].set_ylabel('Feature')
        
        # Plot XGB feature importances
        sns.barplot(x='importance', y='feature', data=xgb_features, ax=axes[1])
        axes[1].set_title(f'XGBoost - Top 15 Features ({model_type.capitalize()})')
        axes[1].set_xlabel('Importance')
        axes[1].set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'{model_type}_feature_importances.png'), dpi=300)
        plt.close()
        print(f"Saved {model_type} feature importances visualization")

def visualize_draft_probabilities(predictions_df):
    """Visualize draft probabilities for players."""
    if predictions_df is None:
        print("No prediction data available for visualization")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create a histogram of draft probabilities
    sns.histplot(predictions_df['pred_draft_prob'], bins=20, kde=True)
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Predicted Draft Probability')
    plt.ylabel('Count')
    plt.title('Distribution of QB Draft Probabilities')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'draft_probabilities.png'), dpi=300)
    plt.close()
    print("Saved draft probabilities visualization")

def visualize_career_status_predictions(predictions_df):
    """Visualize predictions by career status."""
    if predictions_df is None:
        print("No prediction data available for visualization")
        return
    
    # Filter to include only players with career status information
    df = predictions_df[predictions_df['career_status'].notna()].copy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Group by career status and calculate mean values
    status_groups = df.groupby('career_status')
    
    # Plot average wAV by career status
    status_wav = status_groups[['actual_wav', 'pred_wav']].mean().reset_index()
    status_wav = status_wav.melt(id_vars='career_status', 
                               value_vars=['actual_wav', 'pred_wav'],
                               var_name='Metric', value_name='wAV')
    status_wav['Metric'] = status_wav['Metric'].map({'actual_wav': 'Actual wAV', 'pred_wav': 'Predicted wAV'})
    
    sns.barplot(x='career_status', y='wAV', hue='Metric', data=status_wav, ax=axes[0])
    axes[0].set_title('Average wAV by Career Status')
    axes[0].set_xlabel('Career Status')
    axes[0].set_ylabel('wAV')
    
    # Plot average wAV per game by career status
    status_eff = status_groups[['actual_wav_per_game', 'pred_wav_per_game']].mean().reset_index()
    status_eff = status_eff.melt(id_vars='career_status', 
                               value_vars=['actual_wav_per_game', 'pred_wav_per_game'],
                               var_name='Metric', value_name='wAV/Game')
    status_eff['Metric'] = status_eff['Metric'].map({'actual_wav_per_game': 'Actual wAV/Game', 
                                                   'pred_wav_per_game': 'Predicted wAV/Game'})
    
    sns.barplot(x='career_status', y='wAV/Game', hue='Metric', data=status_eff, ax=axes[1])
    axes[1].set_title('Average wAV/Game by Career Status')
    axes[1].set_xlabel('Career Status')
    axes[1].set_ylabel('wAV/Game')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'career_status_predictions.png'), dpi=300)
    plt.close()
    print("Saved career status predictions visualization")

def visualize_draft_confusion_matrix(predictions_df):
    """Create a confusion matrix for draft predictions.
    
    This function visualizes how well the model predicts whether a quarterback
    will be drafted or not by creating a confusion matrix of actual vs. predicted
    draft status. It also identifies and lists the misclassified quarterbacks.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing prediction results
    """
    if predictions_df is None:
        print("No prediction data available for confusion matrix")
        return
    
    # Filter to include only players with actual draft information
    # (excluding active college players without draft status yet)
    df = predictions_df[predictions_df['actual_drafted'].notna()].copy()
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_true = df['actual_drafted'].astype(int)
    y_pred = df['pred_drafted'].astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Drafted', 'Drafted'])
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    
    # Add title and metrics
    plt.title('QB Draft Status Prediction Confusion Matrix', fontsize=16)
    plt.text(0.5, -0.15, 
             f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
             horizontalalignment='center', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'draft_confusion_matrix.png'), dpi=300)
    plt.close()
    print("Saved draft prediction confusion matrix")
    
    # Identify misclassified quarterbacks
    df['correct_prediction'] = y_true == y_pred
    
    # False Positives: Predicted drafted but actually not drafted
    false_positives = df[(df['actual_drafted'] == 0) & (df['pred_drafted'] == 1)]
    
    # False Negatives: Predicted not drafted but actually drafted
    false_negatives = df[(df['actual_drafted'] == 1) & (df['pred_drafted'] == 0)]
    
    # Create a detailed report of misclassified QBs
    misclassified_report_path = os.path.join(VISUALIZATION_DIR, 'misclassified_qbs.csv')
    
    # Combine false positives and false negatives with a classification column
    if not false_positives.empty or not false_negatives.empty:
        false_positives['misclassification_type'] = 'False Positive (Predicted Drafted, Actually Not)'
        false_negatives['misclassification_type'] = 'False Negative (Predicted Not Drafted, Actually Drafted)'
        
        misclassified = pd.concat([false_positives, false_negatives])
        
        # Sort by name for easier reading
        misclassified = misclassified.sort_values('name')
        
        # Select relevant columns for the report
        columns = ['name', 'misclassification_type', 'pred_draft_prob', 
                  'actual_wav', 'actual_wav_per_game', 'pred_wav', 'pred_wav_per_game']
        
        misclassified[columns].to_csv(misclassified_report_path, index=False)
        
        print(f"\nMisclassified quarterbacks:")
        print(f"- False Positives (predicted drafted but weren't): {len(false_positives)}")
        print(f"- False Negatives (not predicted drafted but were): {len(false_negatives)}")
        print(f"Detailed report saved to: {misclassified_report_path}")
        
        # Print the misclassified QBs to console for immediate review
        print("\nFalse Positives (Predicted Drafted but Actually Not):")
        if false_positives.empty:
            print("None")
        else:
            for _, row in false_positives.iterrows():
                print(f"  - {row['name']} (Draft Prob: {row['pred_draft_prob']:.3f})")
        
        print("\nFalse Negatives (Predicted Not Drafted but Actually Were):")
        if false_negatives.empty:
            print("None")
        else:
            for _, row in false_negatives.iterrows():
                print(f"  - {row['name']} (Draft Prob: {row['pred_draft_prob']:.3f})")
    else:
        print("No misclassified quarterbacks found!")

def create_dashboard(data):
    """Create a comprehensive dashboard of all visualizations."""
    print("\nCreating QB Model Results Dashboard...")
    
    # Create individual visualizations
    visualize_model_metrics(data.get('evaluation'))
    visualize_predictions(data.get('predictions'))
    visualize_feature_importances(data.get('feature_importances'))
    visualize_draft_probabilities(data.get('predictions'))
    visualize_career_status_predictions(data.get('predictions'))
    visualize_draft_confusion_matrix(data.get('predictions'))
    
    print("\nAll visualizations saved to:", VISUALIZATION_DIR)

def main():
    """Main function to run all visualizations."""
    print("=== Quarterback Model Results Visualization ===\n")
    
    # Load all necessary data
    data = load_data()
    
    # Create visualizations
    create_dashboard(data)
    
    print("\n=== Visualization Complete ===")

if __name__ == "__main__":
    main()
