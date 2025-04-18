#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quarterback Draft Success Prediction

This used to be two separate files but I merged them since we only needed
the advanced/multi-stage approach. Much cleaner this way.

Basically this handles the QB-specific feature engineering and uses the
core MultiStageModel to do the prediction part. The three stages are:

1. Will they get drafted? (classification)
2. How good will they be? (regression on wAV)
3. How efficient? (regression on wAV/game)
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Use absolute imports - cleaner than relative ones
from src.models.base_model import MultiStageModel  # this has the core modeling logic

# Set up data directory - where we keep the processed data
DATA_DIR = os.path.join(project_root, 'data', 'processed')  # processed = ready for modeling

class NFLDraftQuarterbackModel:
    """
    QB-specific model that predicts draft outcomes and NFL success.
    
    This class handles the quarterback-specific feature extraction
    and passes those features to the MultiStageModel to do the
    actual prediction work. 
    
    Main QB features include: passing metrics, rushing ability,
    team strength of schedule, and combine measurements.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """Set up the QB model with configurable test size and random seed"""
        # basic setup
        self.position_group = "QB"   # only care about quarterbacks
        self.test_size = test_size  # usually use 0.2 = 20% test
        self.random_state = random_state  # for reproducibility
        
        # create the actual model - just pass ourselves as the feature extractor
        # since we have the _extract_player_features method the model needs
        self.model = MultiStageModel(
            position="QB",
            feature_extractor=self,  # we extract features, model does predictions
            random_state=random_state
        )
        
    def prepare_data(self, player_db, target_metric='wAV'):
        """
        Get QB data ready for modeling - filter by position,
        extract features, and get our target values
        """
        # first, filter to just QBs
        qbs = {}
        for pid, pdata in player_db.items():
            # find players marked as QB position
            if 'position' in pdata.get('player_info', {}) and pdata['player_info']['position'] == self.position_group:
                qbs[pid] = pdata
        
        # next, remove guys still in college (no NFL results yet)
        nfl_qbs = {}
        for pid, pdata in qbs.items():
            if pdata['career_status'] != 'active_college':
                nfl_qbs[pid] = pdata
        
        print(f"Found {len(qbs)} {self.position_group} players, {len(nfl_qbs)} with NFL data")
        
        # now extract features for each remaining QB
        print(f"Extracting {self.position_group} features...")
        features = []  # will hold feature dicts
        targets = []   # will hold target values
        ids = []       # will hold player IDs
        
        # loop through QBs and extract features
        for pid, pdata in nfl_qbs.items():
            qb_feats = self._extract_player_features(pid, pdata)
            
            # only use QBs where we could extract features
            if qb_feats:  # skip if None (not enough data)
                # Set target - 0 by default
                tgt = 0
                
                # Check if they have the target metric
                if pdata['draft_info'] and target_metric in pdata['draft_info']:
                    tgt = pdata['draft_info'][target_metric]
                
                # add to our lists
                features.append(qb_feats)
                targets.append(tgt)
                ids.append(pid)
        
        # pandas makes life easier
        X = pd.DataFrame(features)
        y = pd.Series(targets)
        
        print(f"Extracted features from {len(X)} quarterback players")
        return X, y, ids
        
    def _extract_player_features(self, player_id, player_data):
        """
        Pull out QB-specific college stats that might predict NFL success
        
        This is where the position-specific knowledge comes in - we know
        what stats matter for QBs vs other positions
        """
        # start with empty dict to collect features
        features = {}
        
        # Get college stats
        college_stats = player_data.get('college_stats', {})
        if not college_stats:
            return None  # Skip players with no college stats
        
        # Sort seasons in descending order (most recent first)
        # Note: The years are stored as integer keys, not strings
        seasons = sorted([year for year in college_stats.keys() if isinstance(year, (int, str))], 
                         reverse=True)
        
        if not seasons:
            return None  # Skip if no valid seasons
        
        # Only use the two most recent seasons
        recent_seasons = seasons[:2]
        
        # Process seasons
        for i, season in enumerate(recent_seasons):
            # Access using the original key type (integer)
            if season in college_stats:
                season_data = college_stats[season]
                season_suffix = f"_season_{i+1}"  # season_1 is most recent, season_2 is second most recent
                
                # Extract passing stats
                passing_data = season_data.get('passing', {})
                if passing_data:  # Check if not empty
                    for key, value in passing_data.items():
                        # Skip metadata fields
                        if key in ['player', 'player_id', 'position', 'team_name', 'franchise_id', 'season_year']:
                            continue
                        features[f'passing_{key}{season_suffix}'] = value
                
                # Extract rushing stats
                rushing_data = season_data.get('rushing', {})
                if rushing_data:  # Check if not empty
                    for key, value in rushing_data.items():
                        # Skip metadata fields
                        if key in ['player', 'player_id', 'position', 'team_name', 'franchise_id', 'season_year']:
                            continue
                        features[f'rushing_{key}{season_suffix}'] = value
                
                # Extract strength of schedule data
                sos_data = season_data.get('sos_data', {})
                if sos_data:  # Check if not empty
                    for key, value in sos_data.items():
                        # Skip non-numeric fields like 'source'
                        if key == 'source':
                            continue
                        features[f'sos_{key}{season_suffix}'] = value
        
        # Add combine data
        combine_data = player_data.get('combine_stats', {})
        if combine_data:  # Check if not empty
            for key, value in combine_data.items():
                # Skip metadata fields
                if key in ['Player', 'Pos', 'School', 'Drafted (tm/rnd/yr)', 'combine_year', 'scrape_date']:
                    continue
                features[f'combine_{key}'] = value
        
        # Only proceed if we found a reasonable number of features
        if len(features) < 5:  # Arbitrary threshold, adjust as needed
            return None
            
        return features
        
    def train(self, data_path=None, output_dir=None):
        """Run the full training pipeline - load data, train model, save results"""
        # convenience function - wraps everything together
        # Use default paths if not provided
        if data_path is None:
            data_path = os.path.join(DATA_DIR, 'player_database.pkl')
            
            # Fall back to sample database if full one doesn't exist
            if not os.path.exists(data_path):
                sample_path = os.path.join(DATA_DIR, 'player_database_sample.pkl')
                if os.path.exists(sample_path):
                    print(f"Full database not found. Using sample database.")
                    data_path = sample_path
        
        # Load player database
        print(f"Loading player database from: {data_path}")
        try:
            with open(data_path, 'rb') as f:
                player_db = pickle.load(f)
            print(f"Successfully loaded database with {len(player_db)} players")
        except Exception as e:
            print(f"Error loading database: {e}")
            return None
        
        # Run the integrated pipeline
        results = self.model.integrated_pipeline(player_db)
        
        return results
    
    def cross_validate(self, data_path=None, n_folds=10):
        """Run k-fold CV to get more reliable performance metrics
        
        10 folds seems to be the sweet spot - 5 is too few, 20 is overkill
        """
        # cross-validation gives better performance estimates
        # Use default paths if not provided
        if data_path is None:
            data_path = os.path.join(DATA_DIR, 'player_database.pkl')
            
            # Fall back to sample database if full one doesn't exist
            if not os.path.exists(data_path):
                sample_path = os.path.join(DATA_DIR, 'player_database_sample.pkl')
                if os.path.exists(sample_path):
                    print(f"Full database not found. Using sample database.")
                    data_path = sample_path
        
        # Load player database
        print(f"Loading player database from: {data_path}")
        try:
            with open(data_path, 'rb') as f:
                player_db = pickle.load(f)
            print(f"Successfully loaded database with {len(player_db)} players")
        except Exception as e:
            print(f"Error loading database: {e}")
            return None
        
        # Run k-fold cross-validation
        cv_results = self.model.k_fold_cross_validation(player_db, n_folds=n_folds)
        
        return cv_results


def main():
    """Script entry point - run the QB prediction pipeline"""
    print("=== NFL Quarterback Draft Success Prediction Model ===\n")
    
    # create model with fixed random seed
    qb_model = NFLDraftQuarterbackModel(random_state=42)  # always use same seed for reproducibility 
    
    # cross-validation gives better estimate of performance
    # than a single train/test split
    cv_results = qb_model.cross_validate(n_folds=10)  # 10-fold CV
    
    # Save predictions to CSV
    if 'all_predictions' in cv_results:
        predictions_dir = os.path.join(qb_model.model.predictions_dir)
        os.makedirs(predictions_dir, exist_ok=True)
        
        prediction_df = pd.DataFrame(cv_results['all_predictions'])
        prediction_df = prediction_df.sort_values(by='pred_wav', ascending=False)
        prediction_path = os.path.join(predictions_dir, f"qb_predictions.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"\nSaved predictions to {prediction_path}")
    
    # Save evaluation results
    if 'cv_results' in cv_results:
        # Calculate average metrics
        classification_metrics = {}
        regression_metrics = {}
        efficiency_metrics = {}
        
        for model_name in ['rf', 'xgb']:
            # Classification metrics
            if model_name in cv_results['cv_results']['classification']:
                metrics = cv_results['cv_results']['classification'][model_name]
                classification_metrics[model_name] = {
                    'accuracy_mean': float(np.mean(metrics['accuracy'])),
                    'accuracy_std': float(np.std(metrics['accuracy'])),
                    'auc_mean': float(np.mean(metrics['auc'])),
                    'auc_std': float(np.std(metrics['auc']))
                }
            
            # Regression metrics
            if model_name in cv_results['cv_results']['regression']:
                metrics = cv_results['cv_results']['regression'][model_name]
                regression_metrics[model_name] = {
                    'rmse_mean': float(np.mean(metrics['rmse'])),
                    'rmse_std': float(np.std(metrics['rmse'])),
                    'r2_mean': float(np.mean(metrics['r2'])),
                    'r2_std': float(np.std(metrics['r2']))
                }
            
            # Efficiency metrics
            if model_name in cv_results['cv_results']['efficiency']:
                metrics = cv_results['cv_results']['efficiency'][model_name]
                efficiency_metrics[model_name] = {
                    'rmse_mean': float(np.mean(metrics['rmse'])),
                    'rmse_std': float(np.std(metrics['rmse'])),
                    'r2_mean': float(np.mean(metrics['r2'])),
                    'r2_std': float(np.std(metrics['r2']))
                }
        
        # Save to evaluation_results.json
        eval_results = {
            'classification': classification_metrics,
            'regression': regression_metrics,
            'efficiency': efficiency_metrics
        }
        
        output_dir = os.path.join(qb_model.model.output_dir, 'latest')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Saved evaluation results to {output_dir}/evaluation_results.json")
    
    print("\n=== Model Training Completed ===")
    
    # might want to do something with these results later
    return cv_results


if __name__ == "__main__":
    main()
