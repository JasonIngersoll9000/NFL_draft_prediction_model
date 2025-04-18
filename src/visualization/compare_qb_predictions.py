#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script to compare quarterback model predictions with draft expectations.

This script compares three key metrics for quarterbacks:
1. Predicted wAV/game from our quarterback model
2. Expected wAV/game based on draft position (from draft value analysis)
3. Actual wAV/game achieved in the NFL
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Define directories
QB_PREDICTIONS_PATH = os.path.join(project_root, 'results', 'quarterback', 'predictions', 'qb_predictions.csv')
DRAFT_VALUE_PATH = os.path.join(project_root, 'results', 'draft_analysis', 'draft_value_curve.csv')
VISUALIZATION_DIR = os.path.join(project_root, 'results', 'qb_analysis')

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

def load_data():
    """Load QB predictions and draft value data."""
    # Load QB predictions
    qb_predictions = pd.read_csv(QB_PREDICTIONS_PATH)
    print(f"Loaded QB predictions for {len(qb_predictions)} quarterbacks")
    
    # Load draft value curve
    draft_value = pd.read_csv(DRAFT_VALUE_PATH)
    print(f"Loaded draft value data for {len(draft_value)} picks")
    
    return qb_predictions, draft_value

def prepare_draft_value(draft_value):
    """Prepare draft value data for comparison."""
    # Rename column for consistency
    if 'wav_per_game' in draft_value.columns:
        # New format with actual wAV/game values
        draft_value['expected_wav_per_game'] = draft_value['wav_per_game']
    else:
        # Old format with normalized values
        # Get the maximum value from the first pick
        max_value = draft_value.loc[0, 'value']
        
        # Calculate the scaling factor to match typical QB wAV/game values
        # Assuming top QB wAV/game is around 0.9
        scaling_factor = 0.9 / (max_value / 100)
        
        # Apply scaling to all values
        draft_value['expected_wav_per_game'] = draft_value['value'] * scaling_factor
    
    return draft_value

def merge_qb_and_draft_data(qb_predictions, draft_value):
    """Merge QB predictions with draft value data."""
    # Filter to only drafted QBs with NFL experience
    drafted_qbs = qb_predictions[
        (qb_predictions['actual_drafted'] == 1) & 
        (qb_predictions['career_status'] == 'nfl')
    ].copy()
    
    # Add draft pick information (assuming this is not in the predictions file)
    # We'll need to extract this from another source or manually add it
    # For now, let's create a placeholder and we can update it later
    
    # Merge with draft value data
    # We need to know the actual draft pick for each QB to do this properly
    # For now, let's create a simplified version
    
    # Sort by actual wAV/game for visualization
    drafted_qbs = drafted_qbs.sort_values('actual_wav_per_game', ascending=False)
    
    return drafted_qbs

def visualize_qb_value_comparison(qbs_df, draft_value):
    """Create visualizations comparing QB model predictions with draft expectations."""
    # Filter to QBs with actual NFL experience and positive games played
    nfl_qbs = qbs_df[
        (qbs_df['career_status'] == 'nfl') & 
        (qbs_df['actual_wav_per_game'] > 0)
    ].copy()
    
    # Sort by actual wAV/game
    nfl_qbs = nfl_qbs.sort_values('actual_wav_per_game', ascending=False)
    
    # Take top 30 QBs for better visualization
    top_qbs = nfl_qbs.head(30)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot actual wAV/game
    plt.bar(top_qbs['name'], top_qbs['actual_wav_per_game'], 
            alpha=0.7, color='green', label='Actual wAV/game')
    
    # Plot predicted wAV/game
    plt.plot(top_qbs['name'], top_qbs['pred_wav_per_game'], 
             'ro-', markersize=8, label='Predicted wAV/game')
    
    # Add labels and title
    plt.xlabel('Quarterback')
    plt.ylabel('wAV per Game')
    plt.title('Top 30 QBs: Actual vs. Predicted wAV per Game')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'top_qbs_actual_vs_predicted.png'), dpi=300)
    plt.close()
    print("Saved top QBs actual vs. predicted visualization")
    
    # Create scatter plot of actual vs. predicted wAV/game
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation
    correlation = nfl_qbs['actual_wav_per_game'].corr(nfl_qbs['pred_wav_per_game'])
    
    # Create scatter plot
    plt.scatter(nfl_qbs['actual_wav_per_game'], nfl_qbs['pred_wav_per_game'], 
                alpha=0.7, s=80, c=nfl_qbs['pred_draft_prob'], cmap='viridis')
    
    # Add perfect prediction line
    max_val = max(nfl_qbs['actual_wav_per_game'].max(), nfl_qbs['pred_wav_per_game'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Add regression line
    sns.regplot(x='actual_wav_per_game', y='pred_wav_per_game', data=nfl_qbs, 
                scatter=False, line_kws={'color': 'red'})
    
    # Add labels and title
    plt.xlabel('Actual wAV per Game')
    plt.ylabel('Predicted wAV per Game')
    plt.title(f'QB Model Predictions vs. Actual wAV per Game (Correlation: {correlation:.3f})')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Draft Probability')
    
    # Add annotations for notable QBs
    for i, row in nfl_qbs.iterrows():
        if row['actual_wav_per_game'] > 0.8 or row['pred_wav_per_game'] > 0.7:
            plt.annotate(row['name'], 
                         (row['actual_wav_per_game'], row['pred_wav_per_game']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_actual_vs_predicted_scatter.png'), dpi=300)
    plt.close()
    print("Saved QB actual vs. predicted scatter plot")

def visualize_draft_pick_analysis(qbs_df, draft_value):
    """Create visualizations analyzing QB performance relative to draft pick."""
    # We need to add draft pick information to the QBs dataframe
    # For this example, let's manually add draft picks for some notable QBs
    notable_qbs = {
        'Patrick Mahomes': 10,
        'Lamar Jackson': 32,
        'Joe Burrow': 1,
        'Josh Allen': 7,
        'Dak Prescott': 135,
        'Jared Goff': 1,
        'Baker Mayfield': 1,
        'Kyler Murray': 1,
        'Deshaun Watson': 12,
        'Justin Herbert': 6,
        'Trevor Lawrence': 1,
        'Tua Tagovailoa': 5,
        'Mac Jones': 15,
        'Justin Fields': 11,
        'Zach Wilson': 2,
        'C.J. Stroud': 2,
        'Bryce Young': 1,
        'Anthony Richardson': 4,
        'Will Levis': 33,
        'Brock Purdy': 262
    }
    
    # Create a new dataframe with just the notable QBs
    notable_qbs_df = qbs_df[qbs_df['name'].isin(notable_qbs.keys())].copy()
    
    # Add draft pick information
    notable_qbs_df['draft_pick'] = notable_qbs_df['name'].map(notable_qbs)
    
    # Add expected wAV/game based on draft pick
    notable_qbs_df['expected_wav_per_game'] = notable_qbs_df['draft_pick'].apply(
        lambda pick: draft_value.loc[draft_value['pick'] == pick, 'expected_wav_per_game'].values[0] 
        if pick in draft_value['pick'].values else np.nan
    )
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create bar chart with grouped bars
    bar_width = 0.25
    index = np.arange(len(notable_qbs_df))
    
    # Sort by draft pick
    notable_qbs_df = notable_qbs_df.sort_values('draft_pick')
    
    # Plot bars
    plt.bar(index, notable_qbs_df['actual_wav_per_game'], bar_width, 
            label='Actual wAV/game', color='green', alpha=0.7)
    plt.bar(index + bar_width, notable_qbs_df['pred_wav_per_game'], bar_width, 
            label='Predicted wAV/game', color='red', alpha=0.7)
    plt.bar(index + 2*bar_width, notable_qbs_df['expected_wav_per_game'], bar_width, 
            label='Expected wAV/game (by draft pick)', color='blue', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Quarterback')
    plt.ylabel('wAV per Game')
    plt.title('QB Performance: Actual vs. Predicted vs. Draft Expectation')
    
    # Add x-axis labels
    plt.xticks(index + bar_width, [f"{name} (#{pick})" for name, pick in 
                                  zip(notable_qbs_df['name'], notable_qbs_df['draft_pick'])], 
               rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_performance_vs_draft_expectation.png'), dpi=300)
    plt.close()
    print("Saved QB performance vs. draft expectation visualization")
    
    # Create scatter plot of draft pick vs. actual wAV/game
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(notable_qbs_df['draft_pick'], notable_qbs_df['actual_wav_per_game'], 
                alpha=0.7, s=100, c='green', label='Actual wAV/game')
    
    # Add draft value curve
    plt.plot(draft_value['pick'], draft_value['expected_wav_per_game'], 
             'b-', alpha=0.5, label='Expected wAV/game by pick')
    
    # Add annotations for each QB
    for i, row in notable_qbs_df.iterrows():
        plt.annotate(row['name'], 
                     (row['draft_pick'], row['actual_wav_per_game']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8)
    
    # Add labels and title
    plt.xlabel('Draft Pick')
    plt.ylabel('wAV per Game')
    plt.title('QB Performance Relative to Draft Position')
    
    # Set x-axis to log scale to better visualize the range of draft picks
    plt.xscale('log')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_performance_by_draft_position.png'), dpi=300)
    plt.close()
    print("Saved QB performance by draft position visualization")
    
    # Calculate value above expectation
    notable_qbs_df['value_above_expectation'] = notable_qbs_df['actual_wav_per_game'] - notable_qbs_df['expected_wav_per_game']
    
    # Create bar chart of value above expectation
    plt.figure(figsize=(15, 8))
    
    # Sort by value above expectation
    notable_qbs_df = notable_qbs_df.sort_values('value_above_expectation', ascending=False)
    
    # Create bar chart
    bars = plt.bar(notable_qbs_df['name'], notable_qbs_df['value_above_expectation'], 
                   alpha=0.7, color=notable_qbs_df['value_above_expectation'].apply(
                       lambda x: 'green' if x > 0 else 'red'))
    
    # Add draft pick labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pick = notable_qbs_df.iloc[i]['draft_pick']
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'Pick #{pick}', 
                     ha='center', va='bottom', fontsize=8)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height - 0.05, f'Pick #{pick}', 
                     ha='center', va='top', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Quarterback')
    plt.ylabel('Value Above Expectation (wAV/game)')
    plt.title('QB Value Above Draft Position Expectation')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_value_above_expectation.png'), dpi=300)
    plt.close()
    print("Saved QB value above expectation visualization")

def visualize_model_vs_draft_accuracy(qbs_df, draft_value):
    """Compare the accuracy of the QB model vs. draft position in predicting NFL success."""
    # We need to add draft pick information to the QBs dataframe
    # For this example, let's manually add draft picks for some notable QBs
    notable_qbs = {
        'Patrick Mahomes': 10,
        'Lamar Jackson': 32,
        'Joe Burrow': 1,
        'Josh Allen': 7,
        'Dak Prescott': 135,
        'Jared Goff': 1,
        'Baker Mayfield': 1,
        'Kyler Murray': 1,
        'Deshaun Watson': 12,
        'Justin Herbert': 6,
        'Trevor Lawrence': 1,
        'Tua Tagovailoa': 5,
        'Mac Jones': 15,
        'Justin Fields': 11,
        'Zach Wilson': 2,
        'C.J. Stroud': 2,
        'Bryce Young': 1,
        'Anthony Richardson': 4,
        'Will Levis': 33,
        'Brock Purdy': 262
    }
    
    # Create a new dataframe with just the notable QBs
    notable_qbs_df = qbs_df[qbs_df['name'].isin(notable_qbs.keys())].copy()
    
    # Add draft pick information
    notable_qbs_df['draft_pick'] = notable_qbs_df['name'].map(notable_qbs)
    
    # Add expected wAV/game based on draft pick
    notable_qbs_df['expected_wav_per_game'] = notable_qbs_df['draft_pick'].apply(
        lambda pick: draft_value.loc[draft_value['pick'] == pick, 'expected_wav_per_game'].values[0] 
        if pick in draft_value['pick'].values else np.nan
    )
    
    # Calculate errors
    notable_qbs_df['model_error'] = abs(notable_qbs_df['pred_wav_per_game'] - notable_qbs_df['actual_wav_per_game'])
    notable_qbs_df['draft_error'] = abs(notable_qbs_df['expected_wav_per_game'] - notable_qbs_df['actual_wav_per_game'])
    
    # Calculate which is more accurate
    notable_qbs_df['more_accurate'] = notable_qbs_df.apply(
        lambda row: 'Model' if row['model_error'] < row['draft_error'] else 'Draft Position', 
        axis=1
    )
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot
    scatter = plt.scatter(notable_qbs_df['model_error'], notable_qbs_df['draft_error'], 
                          alpha=0.7, s=100, c=notable_qbs_df['more_accurate'].map({'Model': 'green', 'Draft Position': 'red'}))
    
    # Add diagonal line
    max_error = max(notable_qbs_df['model_error'].max(), notable_qbs_df['draft_error'].max())
    plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5)
    
    # Add annotations for each QB
    for i, row in notable_qbs_df.iterrows():
        plt.annotate(row['name'], 
                     (row['model_error'], row['draft_error']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8)
    
    # Add labels and title
    plt.xlabel('Model Error (|Predicted - Actual| wAV/game)')
    plt.ylabel('Draft Position Error (|Expected - Actual| wAV/game)')
    plt.title('QB Model vs. Draft Position: Prediction Accuracy')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Model More Accurate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Draft Position More Accurate')
    ]
    plt.legend(handles=legend_elements)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_model_vs_draft_accuracy.png'), dpi=300)
    plt.close()
    print("Saved QB model vs. draft accuracy visualization")
    
    # Count how many times each is more accurate
    accuracy_counts = notable_qbs_df['more_accurate'].value_counts()
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.pie(accuracy_counts, labels=accuracy_counts.index, autopct='%1.1f%%', 
            colors=['green', 'red'], startangle=90)
    
    # Add title
    plt.title('Which is More Accurate: QB Model or Draft Position?')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'qb_model_vs_draft_accuracy_pie.png'), dpi=300)
    plt.close()
    print("Saved QB model vs. draft accuracy pie chart")

def main():
    """Main function to run all visualizations."""
    print("=== QB Model vs. Draft Expectations Analysis ===\n")
    
    # Load data
    qb_predictions, draft_value = load_data()
    
    # Prepare draft value data with correct wAV/game values
    draft_value = prepare_draft_value(draft_value)
    
    # Merge QB and draft data
    qbs_with_draft = merge_qb_and_draft_data(qb_predictions, draft_value)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_qb_value_comparison(qb_predictions, draft_value)
    visualize_draft_pick_analysis(qb_predictions, draft_value)
    visualize_model_vs_draft_accuracy(qb_predictions, draft_value)
    
    print("\nAll visualizations saved to:", VISUALIZATION_DIR)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
