#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for draft pick value analysis.

This script analyzes the relationship between draft position and NFL success
by calculating average wAV/game for each draft pick across all positions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import glob

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Define directories
DATA_DIR = os.path.join(project_root, 'data', 'raw', 'nfl_draft')
VISUALIZATION_DIR = os.path.join(project_root, 'results', 'draft_analysis')

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

def load_draft_data():
    """Load and combine all draft data files."""
    # Get all draft CSV files
    draft_files = glob.glob(os.path.join(DATA_DIR, 'draft_*.csv'))
    
    if not draft_files:
        print(f"Error: No draft data files found in {DATA_DIR}")
        return None
    
    # Load and combine all draft files
    all_drafts = []
    for file in draft_files:
        try:
            df = pd.read_csv(file)
            # Extract year from filename
            year = os.path.basename(file).split('_')[1].split('.')[0]
            df['draft_year'] = year
            all_drafts.append(df)
            print(f"Loaded draft data for {year} with {len(df)} players")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_drafts:
        print("Error: Could not load any draft data")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_drafts, ignore_index=True)
    
    # Ensure numeric data types
    numeric_cols = ['Rnd', 'Pick', 'Age', 'wAV', 'G']
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Calculate wAV per game
    combined_df['wAV_per_game'] = combined_df.apply(
        lambda row: row['wAV'] / row['G'] if row['G'] > 0 else 0, 
        axis=1
    )
    
    print(f"Combined data contains {len(combined_df)} players across {len(draft_files)} draft years")
    return combined_df

def exponential_decay(x, a, b, c):
    """Exponential decay function for curve fitting."""
    return a * np.exp(-b * x) + c

def visualize_wav_by_pick(draft_df):
    """Visualize average wAV/game by draft pick."""
    # Group by draft pick and calculate average wAV/game
    pick_groups = draft_df.groupby('Pick').agg({
        'wAV_per_game': ['mean', 'std', 'count'],
        'wAV': ['mean', 'std', 'count'],
        'Player': 'count'
    })
    
    # Flatten the multi-index columns
    pick_groups.columns = ['_'.join(col).strip() for col in pick_groups.columns.values]
    pick_groups = pick_groups.reset_index()
    
    # Filter to picks with at least 5 players for more reliable data
    reliable_picks = pick_groups[pick_groups['Player_count'] >= 5].copy()
    
    # Create figure for wAV/game by pick
    plt.figure(figsize=(15, 8))
    
    # Plot average wAV/game by pick
    plt.scatter(reliable_picks['Pick'], reliable_picks['wAV_per_game_mean'], 
                s=reliable_picks['Player_count']*3, alpha=0.7, 
                c=reliable_picks['wAV_per_game_mean'], cmap='viridis')
    
    # Add error bars for standard deviation
    plt.errorbar(reliable_picks['Pick'], reliable_picks['wAV_per_game_mean'], 
                 yerr=reliable_picks['wAV_per_game_std'], fmt='none', alpha=0.3, color='gray')
    
    # Fit exponential decay curve
    try:
        # Get data for curve fitting
        x_data = reliable_picks['Pick'].values
        y_data = reliable_picks['wAV_per_game_mean'].values
        
        # Initial parameter guess
        p0 = [0.5, 0.01, 0.1]
        
        # Fit curve
        popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=p0, maxfev=10000)
        
        # Generate points for the fitted curve
        x_fit = np.linspace(1, max(x_data), 1000)
        y_fit = exponential_decay(x_fit, *popt)
        
        # Plot fitted curve
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fitted curve: {popt[0]:.3f}*exp(-{popt[1]:.3f}*x) + {popt[2]:.3f}')
        plt.legend()
    except Exception as e:
        print(f"Warning: Could not fit exponential decay curve: {e}")
    
    # Add labels and title
    plt.xlabel('Draft Pick')
    plt.ylabel('Average wAV per Game')
    plt.title('Average wAV per Game by Draft Pick (All Positions)')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Average wAV per Game')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'wav_per_game_by_pick.png'), dpi=300)
    plt.close()
    print("Saved wAV per game by draft pick visualization")
    
    # Create figure for first round picks (1-32)
    first_round = pick_groups[(pick_groups['Pick'] >= 1) & (pick_groups['Pick'] <= 32)].copy()
    
    plt.figure(figsize=(15, 8))
    
    # Create bar chart
    bars = plt.bar(first_round['Pick'], first_round['wAV_per_game_mean'], 
                   yerr=first_round['wAV_per_game_std'], alpha=0.7,
                   color=plt.cm.viridis(first_round['wAV_per_game_mean'] / first_round['wAV_per_game_mean'].max()))
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = first_round.iloc[i]['Player_count']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'n={count}', 
                 ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Add labels and title
    plt.xlabel('Draft Pick')
    plt.ylabel('Average wAV per Game')
    plt.title('Average wAV per Game by First Round Draft Pick (All Positions)')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'first_round_wav_per_game.png'), dpi=300)
    plt.close()
    print("Saved first round wAV per game visualization")

def visualize_position_value(draft_df):
    """Visualize average wAV/game by position and draft round."""
    # Group by position and calculate average wAV/game
    position_groups = draft_df.groupby('Pos').agg({
        'wAV_per_game': ['mean', 'std', 'count'],
        'wAV': ['mean', 'count'],
        'Player': 'count'
    })
    
    # Flatten the multi-index columns
    position_groups.columns = ['_'.join(col).strip() for col in position_groups.columns.values]
    position_groups = position_groups.reset_index()
    
    # Filter to positions with at least 10 players
    common_positions = position_groups[position_groups['Player_count'] >= 10].copy()
    
    # Sort by average wAV/game
    common_positions = common_positions.sort_values('wAV_per_game_mean', ascending=False)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(common_positions['Pos'], common_positions['wAV_per_game_mean'], 
                   yerr=common_positions['wAV_per_game_std'], alpha=0.7,
                   color=plt.cm.viridis(common_positions['wAV_per_game_mean'] / common_positions['wAV_per_game_mean'].max()))
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = common_positions.iloc[i]['Player_count']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'n={count}', 
                 ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('Position')
    plt.ylabel('Average wAV per Game')
    plt.title('Average wAV per Game by Position')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'wav_per_game_by_position.png'), dpi=300)
    plt.close()
    print("Saved wAV per game by position visualization")
    
    # Now create a heatmap of position value by draft round
    # Group by position and draft round
    position_round = draft_df.groupby(['Pos', 'Rnd']).agg({
        'wAV_per_game': 'mean',
        'Player': 'count'
    }).reset_index()
    
    # Filter to common positions and rounds 1-7
    position_round = position_round[
        (position_round['Pos'].isin(common_positions['Pos'])) &
        (position_round['Rnd'] >= 1) &
        (position_round['Rnd'] <= 7) &
        (position_round['Player'] >= 5)  # At least 5 players in each cell
    ]
    
    # Create pivot table
    pivot = position_round.pivot(index='Pos', columns='Rnd', values='wAV_per_game')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f', linewidths=0.5)
    
    # Add labels and title
    plt.xlabel('Draft Round')
    plt.ylabel('Position')
    plt.title('Average wAV per Game by Position and Draft Round')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'position_round_heatmap.png'), dpi=300)
    plt.close()
    print("Saved position by round heatmap visualization")

def visualize_draft_value_curve(draft_df):
    """Create a draft value curve based on wAV/game."""
    # Group by draft pick
    pick_groups = draft_df.groupby('Pick').agg({
        'wAV_per_game': ['mean', 'std', 'count'],
        'Player': 'count'
    })
    
    # Flatten the multi-index columns
    pick_groups.columns = ['_'.join(col).strip() for col in pick_groups.columns.values]
    pick_groups = pick_groups.reset_index()
    
    # Filter to picks with at least 5 players
    reliable_picks = pick_groups[pick_groups['Player_count'] >= 5].copy()
    
    # Normalize wAV/game to a 0-100 scale where pick 1 = 100
    max_wav = reliable_picks.loc[reliable_picks['Pick'] == 1, 'wAV_per_game_mean'].values[0]
    reliable_picks['normalized_value'] = reliable_picks['wAV_per_game_mean'] / max_wav * 100
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot normalized value by pick
    plt.scatter(reliable_picks['Pick'], reliable_picks['normalized_value'], 
                s=reliable_picks['Player_count']*3, alpha=0.7)
    
    # Fit exponential decay curve
    try:
        # Get data for curve fitting
        x_data = reliable_picks['Pick'].values
        y_data = reliable_picks['normalized_value'].values
        
        # Initial parameter guess
        p0 = [100, 0.01, 10]
        
        # Fit curve
        popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=p0, maxfev=10000)
        
        # Generate points for the fitted curve
        x_fit = np.linspace(1, max(x_data), 1000)
        y_fit = exponential_decay(x_fit, *popt)
        
        # Plot fitted curve
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fitted curve: {popt[0]:.1f}*exp(-{popt[1]:.4f}*x) + {popt[2]:.1f}')
        
        # Save the curve parameters to a CSV file
        # Create a DataFrame with both normalized values and actual wAV/game values
        curve_data = pd.DataFrame({
            'pick': range(1, 257),
            'normalized_value': exponential_decay(np.array(range(1, 257)), *popt),
            'wav_per_game': exponential_decay(np.array(range(1, 257)), *popt) * max_wav / 100
        })
        curve_data.to_csv(os.path.join(VISUALIZATION_DIR, 'draft_value_curve.csv'), index=False)
        print("Saved draft value curve data to CSV")
        
        # Also save the raw data for each pick
        pick_data = pick_groups[['Pick', 'wAV_per_game_mean', 'wAV_per_game_std', 'Player_count']]
        pick_data.to_csv(os.path.join(VISUALIZATION_DIR, 'draft_pick_wav_per_game.csv'), index=False)
        print("Saved raw draft pick wAV/game data to CSV")
        
        plt.legend()
    except Exception as e:
        print(f"Warning: Could not fit exponential decay curve: {e}")
    
    # Add labels and title
    plt.xlabel('Draft Pick')
    plt.ylabel('Normalized Value (Pick 1 = 100)')
    plt.title('NFL Draft Value Curve Based on wAV per Game')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'draft_value_curve.png'), dpi=300)
    plt.close()
    print("Saved draft value curve visualization")

def visualize_round_value(draft_df):
    """Visualize average wAV/game by draft round."""
    # Group by draft round
    round_groups = draft_df.groupby('Rnd').agg({
        'wAV_per_game': ['mean', 'std', 'count'],
        'wAV': ['mean', 'std'],
        'Player': 'count'
    })
    
    # Flatten the multi-index columns
    round_groups.columns = ['_'.join(col).strip() for col in round_groups.columns.values]
    round_groups = round_groups.reset_index()
    
    # Filter to rounds 1-7
    round_groups = round_groups[(round_groups['Rnd'] >= 1) & (round_groups['Rnd'] <= 7)]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(round_groups['Rnd'], round_groups['wAV_per_game_mean'], 
                   yerr=round_groups['wAV_per_game_std'], alpha=0.7,
                   color=plt.cm.viridis(round_groups['wAV_per_game_mean'] / round_groups['wAV_per_game_mean'].max()))
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = round_groups.iloc[i]['Player_count']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'n={count}', 
                 ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('Draft Round')
    plt.ylabel('Average wAV per Game')
    plt.title('Average wAV per Game by Draft Round')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'wav_per_game_by_round.png'), dpi=300)
    plt.close()
    print("Saved wAV per game by round visualization")

def visualize_pick_ranges(draft_df):
    """Visualize average wAV/game by pick ranges."""
    # Create pick range bins
    bins = [1, 32, 64, 96, 128, 160, 192, 224, 256]
    labels = ['1-32', '33-64', '65-96', '97-128', '129-160', '161-192', '193-224', '225-256']
    
    # Add pick range column
    draft_df['pick_range'] = pd.cut(draft_df['Pick'], bins=bins, labels=labels, right=True)
    
    # Group by pick range
    range_groups = draft_df.groupby('pick_range').agg({
        'wAV_per_game': ['mean', 'std', 'count'],
        'wAV': ['mean', 'std'],
        'Player': 'count'
    })
    
    # Flatten the multi-index columns
    range_groups.columns = ['_'.join(col).strip() for col in range_groups.columns.values]
    range_groups = range_groups.reset_index()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(range_groups['pick_range'], range_groups['wAV_per_game_mean'], 
                   yerr=range_groups['wAV_per_game_std'], alpha=0.7,
                   color=plt.cm.viridis(range_groups['wAV_per_game_mean'] / range_groups['wAV_per_game_mean'].max()))
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = range_groups.iloc[i]['Player_count']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'n={count}', 
                 ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('Draft Pick Range')
    plt.ylabel('Average wAV per Game')
    plt.title('Average wAV per Game by Draft Pick Range')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'wav_per_game_by_pick_range.png'), dpi=300)
    plt.close()
    print("Saved wAV per game by pick range visualization")

def main():
    """Main function to run all visualizations."""
    print("=== NFL Draft Value Analysis ===\n")
    
    # Load draft data
    draft_df = load_draft_data()
    if draft_df is None:
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_wav_by_pick(draft_df)
    visualize_position_value(draft_df)
    visualize_draft_value_curve(draft_df)
    visualize_round_value(draft_df)
    visualize_pick_ranges(draft_df)
    
    print("\nAll visualizations saved to:", VISUALIZATION_DIR)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
