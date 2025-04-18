#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NFL Draft Success Prediction Model

This was originally two model classes (BaseModel and MultiStageModel) but I
consolidated them since we only needed the multi-stage functionality.

Three-stage model for NFL draft success prediction:
- Classification: Predict whether a player will be drafted
- Regression: Predict career wAV for drafted players
- Efficiency: Calculate wAV per game played (efficiency metric)

Works with any position as long as the position-specific feature extractor
implements prepare_data and _extract_player_features methods.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn imports - always seem to need more of these as the project grows
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score  # for regression metrics
import xgboost as xgb  # much better than sklearn's GBM implementation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer  # for handling missing values

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Setup directory structure - makes paths easier to work with
BASE_DIR = project_root  # shorthand
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')  # where all our processed data lives

class MultiStageModel:
    """
    Three-stage model for any position group:
    1. Predicts if a player will get drafted (binary classification) 
    2. For drafted players, predicts career wAV (regression)
    3. For drafted players, predicts wAV per game (efficiency metric)
    
    Need to pass this a position-specific feature extractor that knows
    how to pull the right stats for that position.
    """
    
    def __init__(self, position, feature_extractor, test_size=0.2, random_state=42):
        """Initialize the model
        
        Args:
            position: String position code (e.g., 'QB', 'RB', etc.)
            feature_extractor: Position-specific model that implements the feature extraction
            test_size: Portion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        self.position = position.upper()  # standardize position format
        self.position_name = self._get_position_name(position)  # full position name
        self.test_size = test_size
        self.random_state = random_state
        self.feature_extractor = feature_extractor
        
        # Models for each stage
        self.classification_models = {}
        self.regression_models = {}
        self.efficiency_models = {}
        
        # Set up directory structure
        self._setup_directories()
        
        # Define common hyperparameters for models - allows for easier tuning
        self.rf_params = {
            'classification': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            },
            'regression': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
        }
        
        self.xgb_params = {
            'classification': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            },
            'regression': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state
            }
        }
        
    def load_data(self, data_path):
        """Load player database from pickle file"""
        # Just loads the pickle file from build_player_database.py
        try:
            with open(data_path, 'rb') as f:
                db = pickle.load(f)
                
            # should probably add some validation here someday
            # but it's worked fine so far
            return db
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _setup_directories(self):
        """Create the directory structure for this position"""
        # The primary model output directory structure
        self.model_dir = os.path.join(BASE_DIR, 'models', self.position_name.lower())
        # For saved models and training artifacts
        self.output_dir = os.path.join(self.model_dir, 'latest')  

        # Position-specific results directory
        self.results_dir = os.path.join(BASE_DIR, 'results', self.position_name.lower())
        # For feature importances, evaluation metrics and analysis
        self.features_dir = os.path.join(self.results_dir, 'features')
        # For predictions on test/validation data
        self.predictions_dir = os.path.join(self.results_dir, 'predictions')
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
    
    def _get_position_name(self, position_code):
        """Convert position code to full name"""
        position_map = {
            'QB': 'quarterback',
            'RB': 'runningback',
            'WR': 'widereceiver',
            'TE': 'tightend',
            'OL': 'offensiveline',
            'DL': 'defensiveline',
            'LB': 'linebacker',
            'DB': 'defensiveback'
        }
        return position_map.get(position_code.upper(), position_code.lower())
        
    def check_draft_information(self, player_db):
        """Check how many players have draft information"""
        drafted_count = 0
        non_drafted_count = 0
        
        for player_id, player_data in player_db.items():
            # Skip players of other positions
            if player_data.get('player_info', {}).get('position') != self.position:
                continue
                
            # Check if player was drafted
            draft_info = player_data.get('draft_info', {})
            if draft_info and draft_info.get('draft_year'):
                drafted_count += 1
            else:
                non_drafted_count += 1
        
        print(f"{self.position} players with draft information: {drafted_count}")
        print(f"{self.position} players without draft information: {non_drafted_count}")
        print(f"Total {self.position} players: {drafted_count + non_drafted_count}")
        if drafted_count + non_drafted_count > 0:
            print(f"Percentage drafted: {drafted_count/(drafted_count + non_drafted_count)*100:.2f}%")
        else:
            print(f"No {self.position} players found in database")
    
    def prepare_data(self, player_db):
        """
        Extract features & create the 3 target variables we need:
        - y_drafted: binary classification target
        - y_wav: total career wAV
        - y_wav_per_game: efficiency metric (wAV/games played)
        """
        # Extract features - this depends on position group
        print(f"Extracting {self.position} features...")
        X, _, player_ids = self.feature_extractor.prepare_data(player_db, target_metric='wAV')
        
        # Empty series to fill with target values
        y_drafted = pd.Series(0, index=range(len(player_ids)))  # binary 0/1 
        y_wav = pd.Series(0.0, index=range(len(player_ids)))    # float 
        y_wav_per_game = pd.Series(0.0, index=range(len(player_ids)))  # float
        
        nfl_count = 0
        drafted_count = 0
        
        for idx, player_id in enumerate(player_ids):
            player_data = player_db[player_id]
            
            # Check if the player was drafted (either made NFL or was drafted but didn't play)
            if player_data.get('career_status') in ['nfl', 'drafted_no_nfl']:
                y_drafted.iloc[idx] = 1
                drafted_count += 1
                
                if player_data.get('career_status') == 'nfl':
                    nfl_count += 1
                
                # Get wAV from draft info
                draft_info = player_data.get('draft_info', {})
                if isinstance(draft_info, dict):
                    if 'wAV' in draft_info:
                        # Store wAV for regression
                        wav_value = float(draft_info['wAV'])
                        y_wav.iloc[idx] = wav_value
                        
                        # Calculate wAV per game
                        if 'av_per_game' in draft_info:
                            y_wav_per_game.iloc[idx] = float(draft_info['av_per_game'])
                        elif 'G' in draft_info and draft_info['G'] > 0:
                            games = float(draft_info['G'])
                            y_wav_per_game.iloc[idx] = wav_value / games if games > 0 else 0.0
        
        # Don't want to cheat! Make sure no draft data in features
        print("\nVerifying feature data doesn't contain draft information...")
        draft_cols = [col for col in X.columns if 'draft' in col.lower()]
        if draft_cols:
            print(f"WARNING: Found draft columns, removing: {draft_cols}")
            X = X.drop(columns=draft_cols)  # would be cheating to use these!
        else:
            print("No draft-related features found")
            
        # 5. Print info about the data
        total_players = len(player_ids)
        print(f"\nTotal {self.position} samples: {total_players}")
        print(f"Drafted {self.position} players: {drafted_count} ({drafted_count/total_players:.2%})")
        print(f"NFL {self.position} players: {nfl_count} ({nfl_count/total_players:.2%})")
        
        # Calculate average wAV for drafted players
        drafted_mask = y_drafted == 1
        avg_wav = y_wav[drafted_mask].mean()
        avg_wav_per_game = y_wav_per_game[drafted_mask].mean()
        print(f"Average wAV for drafted {self.position} players: {avg_wav:.2f}")
        print(f"Average wAV/game for drafted {self.position} players: {avg_wav_per_game:.4f}")
        
        print(f"Feature matrix shape: {X.shape}")
        
        return X, y_drafted, y_wav, y_wav_per_game, player_ids
    
    def preprocess_features(self, X):
        """
        Handle missing values and convert all features to numeric.
        
        Returns:
            DataFrame with preprocessed features
        """
        # Keep columns with less than 50% NaN values
        good_cols = X.columns[X.isna().mean() < 0.5].tolist()
        X_filtered = X[good_cols]
        
        print(f"Keeping {len(good_cols)} features out of {X.shape[1]} after filtering for NaN values")
        
        # Convert non-numeric columns to numeric
        numeric_X = X_filtered.copy()
        for col in numeric_X.columns:
            if not pd.api.types.is_numeric_dtype(numeric_X[col]):
                try:
                    numeric_X[col] = pd.to_numeric(numeric_X[col], errors='coerce')
                except:
                    numeric_X = numeric_X.drop(columns=[col])
                    print(f"Dropped non-numeric column: {col}")
        
        # Fill remaining NaN values with 0
        numeric_X = numeric_X.fillna(0)
        
        return numeric_X
    
    def train_classification_model(self, X_train, y_train):
        """
        Train classification models to predict if a player will be drafted.
        
        Returns:
            Dictionary of trained classification models
        """
        print("\nTraining draft classification models...")
        
        # Use parameters from the class-level dictionaries
        # Random Forest Classifier
        rf_params = self.rf_params['classification'].copy()
        rf_params['class_weight'] = 'balanced'  # Handle class imbalance
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # XGBoost Classifier
        xgb_params = self.xgb_params['classification'].copy()
        xgb_params['scale_pos_weight'] = sum(y_train == 0) / sum(y_train == 1)  # Balance classes
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        self.classification_models = {
            'rf': rf_model,
            'xgb': xgb_model
        }
        
        return self.classification_models
    
    def train_regression_model(self, X_train, y_train):
        """
        Train regression models to predict wAV for drafted players.
        
        Args:
            X_train: Features for drafted players
            y_train: wAV values for these players
            
        Returns:
            Dictionary of trained regression models
        """
        print(f"\nTraining wAV regression models for {len(X_train)} drafted {self.position} players...")
        
        # Random Forest Regressor - use common parameters
        rf_model = RandomForestRegressor(**self.rf_params['regression'])
        rf_model.fit(X_train, y_train)
        
        # XGBoost Regressor - use common parameters
        xgb_model = xgb.XGBRegressor(**self.xgb_params['regression'])
        xgb_model.fit(X_train, y_train)
        
        self.regression_models = {
            'rf': rf_model,
            'xgb': xgb_model
        }
        
        return self.regression_models
    
    def train_efficiency_model(self, X_train, y_train):
        """
        Train regression models to predict wAV per game for drafted players.
        
        Args:
            X_train: Features for drafted players
            y_train: wAV per game values for these players
            
        Returns:
            Dictionary of trained regression models
        """
        print(f"\nTraining wAV per game regression models for {len(X_train)} drafted {self.position} players...")
        
        # Since efficiency models use the same regression settings as the total wAV models,
        # we can reuse the same parameters
        
        # Random Forest Regressor for wAV per game
        rf_model = RandomForestRegressor(**self.rf_params['regression'])
        rf_model.fit(X_train, y_train)
        
        # XGBoost Regressor for wAV per game
        xgb_model = xgb.XGBRegressor(**self.xgb_params['regression'])
        xgb_model.fit(X_train, y_train)
        
        self.efficiency_models = {
            'rf': rf_model,
            'xgb': xgb_model
        }
        
        return self.efficiency_models
    
    def evaluate_classification_models(self, models, X_train, y_train, X_test, y_test):
        """
        Evaluate classification models and return metrics.
        
        Returns:
            Dictionary with evaluation metrics for each model
        """
        print("\nEvaluating draft classification models...")
        
        results = {}
        
        for name, model in models.items():
            # Training predictions
            y_train_pred = model.predict(X_train)
            y_train_prob = model.predict_proba(X_train)[:, 1]
            
            # Testing predictions
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            train_auc = roc_auc_score(y_train, y_train_prob)
            test_auc = roc_auc_score(y_test, y_test_prob)
            
            # Store results
            results[name] = {
                'train': {
                    'accuracy': train_accuracy,
                    'auc': train_auc,
                    'classification_report': classification_report(y_train, y_train_pred, output_dict=True)
                },
                'test': {
                    'accuracy': test_accuracy,
                    'auc': test_auc,
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
                }
            }
            
            # Print key metrics
            print(f"  {name.upper()} - Training: Accuracy={train_accuracy:.4f}, AUC={train_auc:.4f}")
            print(f"  {name.upper()} - Testing:  Accuracy={test_accuracy:.4f}, AUC={test_auc:.4f}")
            
            # Display confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"  Confusion Matrix (Testing):")
            print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
            print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return results
    
    def evaluate_regression_models(self, models, X_train, y_train, X_test, y_test):
        """
        Evaluate regression models and return metrics.
        
        Returns:
            Dictionary with evaluation metrics for each model
        """
        print("\nEvaluating wAV regression models...")
        
        results = {}
        
        for name, model in models.items():
            # Training predictions
            y_train_pred = model.predict(X_train)
            
            # Testing predictions
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Store results
            results[name] = {
                'train': {
                    'rmse': train_rmse,
                    'r2': train_r2
                },
                'test': {
                    'rmse': test_rmse,
                    'r2': test_r2
                }
            }
            
            # Print key metrics
            print(f"  {name.upper()} - Training: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
            print(f"  {name.upper()} - Testing:  RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
        
        return results
    
    def evaluate_efficiency_models(self, models, X_train, y_train, X_test, y_test):
        """
        Evaluate wAV per game efficiency models and return metrics.
        
        Returns:
            Dictionary with evaluation metrics for each model
        """
        print("\nEvaluating wAV-per-game efficiency models...")
        
        results = {}
        
        for name, model in models.items():
            # Training predictions
            y_train_pred = model.predict(X_train)
            
            # Testing predictions
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Store results
            results[name] = {
                'train': {
                    'rmse': train_rmse,
                    'r2': train_r2
                },
                'test': {
                    'rmse': test_rmse,
                    'r2': test_r2
                }
            }
            
            # Print key metrics
            print(f"  {name.upper()} - Training: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
            print(f"  {name.upper()} - Testing:  RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
        
        return results
    
    def get_feature_importances(self, model_type, models, features):
        """
        Get feature importances from trained models.
        
        Args:
            model_type: 'classification' or 'regression'
            models: Dictionary of trained models
            features: Feature names
            
        Returns:
            Dictionary with feature importances for each model
        """
        importances = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importances[model_name] = {
                    feature: float(imp) for feature, imp in zip(features, importance)
                }
                
                # Print top 10 features
                sorted_imp = sorted(importances[model_name].items(), key=lambda x: x[1], reverse=True)[:10]
                
                print(f"\nTop 10 features for {model_type} - {model_name.upper()}:")
                for feature, imp in sorted_imp:
                    print(f"  {feature}: {imp:.4f}")
        
        return importances
    
    def save_results(self, output_dir, classification_results, regression_results, efficiency_results):
        """Save model results, evaluation metrics, and feature importances"""
        
        model_dir = os.path.join(output_dir, 'latest')  # Use 'latest' instead of timestamp
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.classification_models.items():
            model_path = os.path.join(model_dir, f"classification_{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        for name, model in self.regression_models.items():
            model_path = os.path.join(model_dir, f"regression_{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        for name, model in self.efficiency_models.items():
            model_path = os.path.join(model_dir, f"efficiency_{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save evaluation results
        results = {
            'classification': classification_results,
            'regression': regression_results,
            'efficiency': efficiency_results
        }
        
        results_path = os.path.join(model_dir, "evaluation_results.json")
        
        # Use our class method for serialization
        
        import json
        with open(results_path, 'w') as f:
            json.dump(self.convert_to_serializable(results), f, indent=2)
        
        print(f"Models and results saved to {model_dir}")
        
    def k_fold_cross_validation(self, player_db, n_folds=10):
        """
        Perform k-fold cross-validation to get more reliable model performance metrics.
        
        Args:
            player_db: Player database dictionary
            n_folds: Number of folds for CV (default: 10)
            
        Returns:
            Dictionary with CV results and final models trained on all data
        """
        # 1. Prepare data
        print(f"\n=== Performing {n_folds}-fold Cross-Validation ===")
        print("\nPreparing data...")
        X, y_drafted, y_wav, y_wav_per_game, player_ids = self.prepare_data(player_db)
        
        # 2. Preprocess features
        X_processed = self.preprocess_features(X)
        print(f"Using {X_processed.shape[1]} features")
        
        # 3. Initialize CV splitter
        # Use StratifiedKFold to maintain class balance across folds
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # 4. Initialize result trackers
        cv_results = {
            'classification': {
                'rf': {'accuracy': [], 'auc': [], 'conf_matrix': []},
                'xgb': {'accuracy': [], 'auc': [], 'conf_matrix': []}
            },
            'regression': {
                'rf': {'rmse': [], 'r2': []},
                'xgb': {'rmse': [], 'r2': []}
            },
            'efficiency': {
                'rf': {'rmse': [], 'r2': []},
                'xgb': {'rmse': [], 'r2': []}
            },
            'feature_importance': {
                'classification': {'rf': {}, 'xgb': {}},
                'regression': {'rf': {}, 'xgb': {}},
                'efficiency': {'rf': {}, 'xgb': {}}
            }
        }
        
        # 5. Perform k-fold CV
        print(f"\nStarting {n_folds}-fold cross-validation...")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_processed, y_drafted)):
            print(f"\nFold {fold+1}/{n_folds}")
            
            # Get train/test data for this fold
            X_train = X_processed.iloc[train_idx]
            X_test = X_processed.iloc[test_idx]
            y_train_drafted = y_drafted.iloc[train_idx]
            y_test_drafted = y_drafted.iloc[test_idx]
            
            # Get indices for getting regression targets
            train_original_indices = train_idx
            test_original_indices = test_idx
            
            # Classification stage
            print(f"  Classification - Train: {len(X_train)} samples, {y_train_drafted.mean():.2%} drafted")
            print(f"  Classification - Test: {len(X_test)} samples, {y_test_drafted.mean():.2%} drafted")
            
            # Train classification models
            classification_models = self.train_classification_model(X_train, y_train_drafted)
            
            # Evaluate classification models
            for model_name, model in classification_models.items():
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_drafted, y_pred)
                auc = roc_auc_score(y_test_drafted, y_pred_proba)
                conf = confusion_matrix(y_test_drafted, y_pred)
                
                # Store metrics
                cv_results['classification'][model_name]['accuracy'].append(accuracy)
                cv_results['classification'][model_name]['auc'].append(auc)
                cv_results['classification'][model_name]['conf_matrix'].append(conf)
                
                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(X_processed.columns):
                        if feature not in cv_results['feature_importance']['classification'][model_name]:
                            cv_results['feature_importance']['classification'][model_name][feature] = []
                        cv_results['feature_importance']['classification'][model_name][feature].append(importances[i])
                
                print(f"  {model_name.upper()} - Classification: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
            
            # Filter for drafted players
            train_drafted_indices = np.where(y_train_drafted == 1)[0]
            test_drafted_indices = np.where(y_test_drafted == 1)[0]
            
            # Get regression data
            X_train_drafted = X_train.iloc[train_drafted_indices].reset_index(drop=True)
            X_test_drafted = X_test.iloc[test_drafted_indices].reset_index(drop=True)
            
            # Get original indices for these drafted players
            train_drafted_original = [train_original_indices[i] for i in train_drafted_indices]
            test_drafted_original = [test_original_indices[i] for i in test_drafted_indices]
            
            # Get target values for wAV
            y_wav_array = y_wav.to_numpy()
            y_train_wav = y_wav_array[train_drafted_original]
            y_test_wav = y_wav_array[test_drafted_original]
            
            # Get target values for wAV per game
            y_wav_per_game_array = y_wav_per_game.to_numpy()
            y_train_wav_per_game = y_wav_per_game_array[train_drafted_original]
            y_test_wav_per_game = y_wav_per_game_array[test_drafted_original]
            
            # Print regression data size
            print(f"  Regression - Train: {len(X_train_drafted)} drafted players")
            print(f"  Regression - Test: {len(X_test_drafted)} drafted players")
            
            # Skip regression if too few samples in test set
            if len(X_test_drafted) < 3:
                print("  Skipping regression: Too few drafted players in test set")
                continue
                
            # Train and evaluate wAV regression
            regression_models = self.train_regression_model(X_train_drafted, y_train_wav)
            for model_name, model in regression_models.items():
                # Get predictions
                y_pred = model.predict(X_test_drafted)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test_wav, y_pred))
                r2 = r2_score(y_test_wav, y_pred)
                
                # Store metrics
                cv_results['regression'][model_name]['rmse'].append(rmse)
                cv_results['regression'][model_name]['r2'].append(r2)
                
                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(X_processed.columns):
                        if feature not in cv_results['feature_importance']['regression'][model_name]:
                            cv_results['feature_importance']['regression'][model_name][feature] = []
                        cv_results['feature_importance']['regression'][model_name][feature].append(importances[i])
                
                print(f"  {model_name.upper()} - wAV Regression: RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Train and evaluate wAV per game regression
            efficiency_models = self.train_efficiency_model(X_train_drafted, y_train_wav_per_game)
            for model_name, model in efficiency_models.items():
                # Get predictions
                y_pred = model.predict(X_test_drafted)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test_wav_per_game, y_pred))
                r2 = r2_score(y_test_wav_per_game, y_pred)
                
                # Store metrics
                cv_results['efficiency'][model_name]['rmse'].append(rmse)
                cv_results['efficiency'][model_name]['r2'].append(r2)
                
                # Get feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feature in enumerate(X_processed.columns):
                        if feature not in cv_results['feature_importance']['efficiency'][model_name]:
                            cv_results['feature_importance']['efficiency'][model_name][feature] = []
                        cv_results['feature_importance']['efficiency'][model_name][feature].append(importances[i])
                
                print(f"  {model_name.upper()} - wAV/Game: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # 6. Summarize CV results
        print("\n=== Cross-Validation Results Summary ===")
        
        # Classification results
        print("\nClassification Results:")
        for model_name in ['rf', 'xgb']:
            accuracies = cv_results['classification'][model_name]['accuracy']
            aucs = cv_results['classification'][model_name]['auc']
            print(f"  {model_name.upper()} - Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            print(f"  {model_name.upper()} - AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        
        # Regression results
        print("\nwAV Regression Results:")
        for model_name in ['rf', 'xgb']:
            rmses = cv_results['regression'][model_name]['rmse']
            r2s = cv_results['regression'][model_name]['r2']
            print(f"  {model_name.upper()} - RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"  {model_name.upper()} - R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        
        # Efficiency results
        print("\nwAV/Game Efficiency Results:")
        for model_name in ['rf', 'xgb']:
            rmses = cv_results['efficiency'][model_name]['rmse']
            r2s = cv_results['efficiency'][model_name]['r2']
            print(f"  {model_name.upper()} - RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"  {model_name.upper()} - R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        
        # 7. Get average feature importance
        avg_importances = {
            'classification': {},
            'regression': {},
            'efficiency': {}
        }
        
        for model_type in ['classification', 'regression', 'efficiency']:
            for model_name in ['rf', 'xgb']:
                importances = cv_results['feature_importance'][model_type][model_name]
                if importances:
                    avg_importances[model_type][model_name] = {}
                    for feature, values in importances.items():
                        avg_importances[model_type][model_name][feature] = np.mean(values)
        
        # 8. Train final models on all data
        print("\n=== Training Final Models on All Data ===")
        
        # Classification models
        print("\nTraining final classification models...")
        final_classification_models = self.train_classification_model(X_processed, y_drafted)
        self.classification_models = final_classification_models
        
        # Get drafted player indices for regression
        all_drafted_indices = np.where(y_drafted == 1)[0]
        X_all_drafted = X_processed.iloc[all_drafted_indices].reset_index(drop=True)
        y_all_wav = y_wav.to_numpy()[all_drafted_indices]
        y_all_wav_per_game = y_wav_per_game.to_numpy()[all_drafted_indices]
        
        # Regression models
        print(f"\nTraining final wAV regression models ({len(X_all_drafted)} drafted players)...")
        final_regression_models = self.train_regression_model(X_all_drafted, y_all_wav)
        self.regression_models = final_regression_models
        
        # Efficiency models
        print(f"\nTraining final wAV/game efficiency models ({len(X_all_drafted)} drafted players)...")
        final_efficiency_models = self.train_efficiency_model(X_all_drafted, y_all_wav_per_game)
        self.efficiency_models = final_efficiency_models
        
        # Save top features from CV
        self.save_cv_feature_importances(avg_importances)
        
        # Save the final trained models
        print("\nSaving final trained models...")
        self.save_results(
            self.output_dir, 
            {'rf': {}, 'xgb': {}},  # Placeholder for evaluation metrics
            {'rf': {}, 'xgb': {}}, 
            {'rf': {}, 'xgb': {}}
        )
        print(f"Models saved to {self.output_dir}")
        
        # 9. Generate final predictions on all data
        print("\nGenerating predictions for all QBs...")
        
        # Predict draft status for all QBs
        classifier = self.classification_models['rf']
        y_pred_drafted = classifier.predict(X_processed)
        y_pred_prob = classifier.predict_proba(X_processed)[:, 1]
        
        # Predict wAV and wAV/game for all QBs
        regressor = self.regression_models['rf']
        efficiency = self.efficiency_models['rf']
        y_pred_wav = regressor.predict(X_processed)
        y_pred_wav_per_game = efficiency.predict(X_processed)
        
        # 10. Create prediction results
        all_predictions = []
        for i, player_id in enumerate(player_ids):
            player_data = player_db[player_id]
            player_name = player_data.get('player_info', {}).get('name', 'Unknown')
            career_status = player_data.get('career_status', 'unknown')
            
            actual_drafted = y_drafted.iloc[i]
            
            # Get actual metrics
            actual_wav = 0.0
            actual_wav_per_game = 0.0
            draft_info = player_data.get('draft_info', {})
            if isinstance(draft_info, dict):
                if 'wAV' in draft_info:
                    actual_wav = float(draft_info['wAV'])
                if 'av_per_game' in draft_info:
                    actual_wav_per_game = float(draft_info['av_per_game'])
                elif 'wAV' in draft_info and 'G' in draft_info and draft_info['G'] > 0:
                    games = float(draft_info['G'])
                    actual_wav_per_game = actual_wav / games if games > 0 else 0.0
            
            all_predictions.append({
                'id': player_id,
                'name': player_name,
                'career_status': career_status,
                'actual_drafted': int(actual_drafted),
                'actual_wav': float(actual_wav),
                'actual_wav_per_game': float(actual_wav_per_game),
                'pred_drafted': int(y_pred_drafted[i]),
                'pred_draft_prob': float(y_pred_prob[i]),
                'pred_wav': float(y_pred_wav[i]),
                'pred_wav_per_game': float(y_pred_wav_per_game[i])
            })
        
        # Ensure wAV predictions are 0 for players predicted not to be drafted
        for result in all_predictions:
            if result['pred_drafted'] == 0:
                result['pred_wav'] = 0.0
                result['pred_wav_per_game'] = 0.0
        
        # Sort by predicted wAV
        all_predictions.sort(key=lambda x: x['pred_wav'], reverse=True)
        
        return {
            'cv_results': cv_results,
            'avg_importances': avg_importances,
            'all_predictions': all_predictions
        }
    
    def convert_to_serializable(self, obj):
        """Fix numpy types so JSON doesn't break"""
        # This is super annoying - numpy types aren't JSON serializable
        # Have to convert them all to standard Python types
        if isinstance(obj, (np.integer, np.int32, np.int64)):  # numpy ints
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):   # numpy floats
            return float(obj)
        elif isinstance(obj, np.ndarray):  # arrays -> lists
            return obj.tolist()
        elif isinstance(obj, dict):  # recursive for dicts
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # recursive for lists
            return [self.convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):  # recursive for tuples
            return tuple(self.convert_to_serializable(i) for i in obj)        
        else:
            return obj  # leave other types alone
            
    def run(self, data_path, output_dir=None):
        """Run the whole shebang - load data, train models, get results
        
        This is basically a wrapper that handles the boring stuff and then
        calls integrated_pipeline to do the actual work
        
        Args:
            data_path: Path to player database
            output_dir: Where to save the results
            
        Returns:
            Dictionary of results and models
        """
        # Use default dir if none provided
        if not output_dir:
            output_dir = os.path.join(BASE_DIR, 'results', self.position.lower())
        
        # load player database
        print(f"Loading player database from: {data_path}")
        player_db = self.load_data(data_path)
        if not player_db:
            print("Failed to load database!")
            return None
            
        print(f"Loaded {len(player_db)} players")
            
        # let's do this!
        results = self.integrated_pipeline(player_db)
        
        return results
    
    def display_player_prediction(self, idx, player, include_header=True):
        """Helper function to display player prediction stats with consistent formatting"""
        if include_header:
            print(f"\n{idx+1}. {player['name']}:")
        else:
            print(f"   {player['name']}:")
            
        print(f"  Career Status: {player['career_status']}")
        print(f"  Predicted Draft Probability: {player['pred_draft_prob']:.2f}")
        print(f"  Actual wAV: {player['actual_wav']:.1f}")
        print(f"  Actual wAV/game: {player['actual_wav_per_game']:.4f}")
        print(f"  Predicted wAV: {player['pred_wav']:.1f}")
        print(f"  Predicted wAV/game: {player['pred_wav_per_game']:.4f}")
        
    def save_cv_feature_importances(self, importances):
        """Save cross-validation feature importances to files"""
        # Ensure the output directory exists
        os.makedirs(self.features_dir, exist_ok=True)
        
        # For each model type and model name, save importances
        for model_type, model_importances in importances.items():
            for model_name, features_df in model_importances.items():
                # Check if features_df is a DataFrame or dict
                if isinstance(features_df, dict):
                    # Convert dict to DataFrame
                    features_list = [
                        {'feature': feature, 'importance': importance}
                        for feature, importance in features_df.items()
                    ]
                    sorted_df = pd.DataFrame(features_list).sort_values('importance', ascending=False)
                else:
                    # It's already a DataFrame
                    sorted_df = features_df.sort_values('importance', ascending=False)
                
                # Save to CSV
                output_path = os.path.join(self.features_dir, f"cv_{model_type}_{model_name}_importances.csv")
                sorted_df.to_csv(output_path, index=False)
                print(f"Saved {model_type} {model_name} feature importances to {output_path}")
    
    def integrated_pipeline(self, player_db):
        """Run the full three-stage modeling pipeline"""
        
        # 0. Check how many players have draft information
        self.check_draft_information(player_db)
        
        # 1. Prepare data
        print("Preparing data for three-stage modeling...")
        X, y_drafted, y_wav, y_wav_per_game, player_ids = self.prepare_data(player_db)
        
        # 2. Preprocess features (handle missing values, etc.)
        X_processed = self.preprocess_features(X)
        print(f"Keeping {X_processed.shape[1]} features out of {X.shape[1]} after filtering for NaN values")
        
        # 3. Split data into train/test sets
        X_train, X_test, y_train_drafted, y_test_drafted, train_indices, test_indices = train_test_split(
            X_processed, y_drafted, np.arange(len(X_processed)), test_size=0.2, random_state=self.random_state,
            stratify=y_drafted
        )
        
        # Get corresponding player IDs for train/test sets
        ids_train = [player_ids[i] for i in train_indices]
        ids_test = [player_ids[i] for i in test_indices]
        
        print("\nClassification split:")
        print(f"  Training samples: {len(X_train)} (Drafted rate: {y_train_drafted.mean():.2%})")
        print(f"  Testing samples: {len(X_test)} (Drafted rate: {y_test_drafted.mean():.2%})")
        
        # 4. Train and evaluate classification models
        classification_models = self.train_classification_model(X_train, y_train_drafted)
        classification_results = self.evaluate_classification_models(
            classification_models, X_train, y_train_drafted, X_test, y_test_drafted
        )
        classification_importances = self.get_feature_importances(
            'classification', classification_models, X_processed.columns
        )
        
        # 5. Filter for drafted players for the regression steps
        y_wav_array = y_wav.to_numpy()
        y_wav_per_game_array = y_wav_per_game.to_numpy()
        
        # Find indices of drafted players in train and test sets
        train_drafted_indices = np.where(y_train_drafted == 1)[0]
        test_drafted_indices = np.where(y_test_drafted == 1)[0]
        
        # Filter features for drafted players
        X_train_drafted = X_train.iloc[train_drafted_indices].reset_index(drop=True)
        X_test_drafted = X_test.iloc[test_drafted_indices].reset_index(drop=True)
        
        # Get the original indices for these drafted players
        train_original_indices = [train_indices[i] for i in train_drafted_indices]
        test_original_indices = [test_indices[i] for i in test_drafted_indices]
        
        # Get target values for drafted players
        y_train_wav = y_wav_array[train_original_indices]
        y_test_wav = y_wav_array[test_original_indices]
        
        y_train_wav_per_game = y_wav_per_game_array[train_original_indices]
        y_test_wav_per_game = y_wav_per_game_array[test_original_indices]
        
        print(f"\nRegression split (only drafted QBs):")
        print(f"  Training samples: {len(X_train_drafted)} (Mean wAV: {np.mean(y_train_wav):.2f})")
        print(f"  Testing samples: {len(X_test_drafted)} (Mean wAV: {np.mean(y_test_wav):.2f})")
        print(f"  Training samples wAV/game: {np.mean(y_train_wav_per_game):.4f}")
        print(f"  Testing samples wAV/game: {np.mean(y_test_wav_per_game):.4f}")
        
        # 6. Train and evaluate total wAV regression models
        regression_models = self.train_regression_model(X_train_drafted, y_train_wav)
        regression_results = self.evaluate_regression_models(
            regression_models, X_train_drafted, y_train_wav, X_test_drafted, y_test_wav
        )
        regression_importances = self.get_feature_importances(
            'regression', regression_models, X_processed.columns
        )
        
        # 7. Train and evaluate wAV per game efficiency models
        efficiency_models = self.train_efficiency_model(X_train_drafted, y_train_wav_per_game)
        efficiency_results = self.evaluate_efficiency_models(
            efficiency_models, X_train_drafted, y_train_wav_per_game, X_test_drafted, y_test_wav_per_game
        )
        efficiency_importances = self.get_feature_importances(
            'efficiency', efficiency_models, X_processed.columns
        )
        
        # 8. Save all results
        self.save_results(self.output_dir, classification_results, regression_results, efficiency_results)
        
        # 9. Return a sample of player predictions
        sample_predictions = self.generate_sample_predictions(
            player_db, ids_test, X_test, y_test_drafted, y_wav, y_wav_per_game
        )
        
        return {
            'classification_results': classification_results,
            'regression_results': regression_results,
            'efficiency_results': efficiency_results,
            'sample_predictions': sample_predictions
        }
    
    def generate_sample_predictions(self, player_db, player_ids, X_test, y_true_drafted, y_wav, y_wav_per_game):
        """Generate predictions for a sample of players in the test set"""
        
        results = []
        
        # Get player information
        for i, player_id in enumerate(player_ids):
            player = player_db[player_id]
            
            # Get actual draft status
            career_status = player.get('career_status', 'unknown')
            actual_drafted = 1 if career_status in ['nfl', 'drafted_no_nfl'] else 0
            
            # Make predictions
            draft_probs = {}
            for model_name, model in self.classification_models.items():
                if hasattr(model, 'predict_proba'):
                    draft_probs[model_name] = model.predict_proba(X_test.iloc[[i]])[:, 1][0]
                else:
                    draft_probs[model_name] = float(model.predict(X_test.iloc[[i]])[0])
            
            # Average probabilities from different models
            avg_draft_prob = sum(draft_probs.values()) / len(draft_probs)
            pred_drafted = 1 if avg_draft_prob >= 0.5 else 0
            
            # For players predicted NOT to be drafted, set wAV values to zero
            if pred_drafted == 0:
                pred_wav = 0.0
                pred_wav_per_game = 0.0
            else:
                # Regression predictions (wAV)
                wav_preds = {}
                for model_name, model in self.regression_models.items():
                    wav_preds[model_name] = float(model.predict(X_test.iloc[[i]])[0])
                pred_wav = sum(wav_preds.values()) / len(wav_preds)
                
                # Efficiency predictions (wAV per game)
                wav_per_game_preds = {}
                for model_name, model in self.efficiency_models.items():
                    wav_per_game_preds[model_name] = float(model.predict(X_test.iloc[[i]])[0])
                pred_wav_per_game = sum(wav_per_game_preds.values()) / len(wav_per_game_preds)
            
            # Get actual values
            actual_wav = y_wav.iloc[i] if i < len(y_wav) else 0
            actual_wav_per_game = y_wav_per_game.iloc[i] if i < len(y_wav_per_game) else 0
            
            results.append({
                'id': player_id,
                'name': player.get('player_info', {}).get('name', 'Unknown'),
                'career_status': career_status,
                'actual_drafted': int(actual_drafted),
                'actual_wav': float(actual_wav),
                'actual_wav_per_game': float(actual_wav_per_game),
                'pred_drafted': int(pred_drafted),
                'pred_draft_prob': float(avg_draft_prob),
                'pred_wav': float(pred_wav),
                'pred_wav_per_game': float(pred_wav_per_game)
            })
        
        # Ensure wAV predictions are 0 for players predicted not to be drafted
        for result in results:
            if result['pred_drafted'] == 0:
                result['pred_wav'] = 0.0
                result['pred_wav_per_game'] = 0.0
        
        # Sort by predicted wAV
        results.sort(key=lambda x: x['pred_wav'], reverse=True)
        
        return results  # Return ALL players, not just a subset


def main():
    print("=== Multi-Stage Prediction Model with 10-Fold CV ===\n")
    
    # Load player database
    print("Loading player database...")
    try:
        with open(os.path.join(DATA_DIR, 'player_database.pkl'), 'rb') as f:
            player_db = pickle.load(f)
        print(f"Successfully loaded database with {len(player_db)} players")
    except Exception as e:
        print(f"Error loading database: {e}")
        return
    
    # Create and run model with cross-validation
    model = MultiStageModel(random_state=42)
    
    # Check draft information
    model.check_draft_information(player_db)
    
    # Run k-fold cross-validation
    cv_results = model.k_fold_cross_validation(player_db, n_folds=10)
    
    # Get all predictions
    predictions = cv_results['all_predictions']
    
    # Categorize predictions
    false_negatives = [p for p in predictions if p['actual_drafted'] == 1 and p['pred_drafted'] == 0]
    false_positives = [p for p in predictions if p['actual_drafted'] == 0 and p['pred_drafted'] == 1]
    true_positives = [p for p in predictions if p['actual_drafted'] == 1 and p['pred_drafted'] == 1]
    true_negatives = [p for p in predictions if p['actual_drafted'] == 0 and p['pred_drafted'] == 0]
    
    # Calculate overall metrics
    total = len(predictions)
    accuracy = (len(true_positives) + len(true_negatives)) / total
    
    # Display sample predictions
    print("\n=== Sample Player Predictions ===")
    
    # Print False Negatives (Players incorrectly predicted to not be drafted)
    if false_negatives:
        print("\n--- PLAYERS WHO WERE DRAFTED BUT PREDICTION MISSED ---")
        for i, player in enumerate(sorted(false_negatives, key=lambda x: x['actual_wav'], reverse=True)[:10]):
            draft_model.display_player_prediction(i, player)
    
    # Print a few correctly predicted drafted players
    if true_positives:
        print("\n--- TOP CORRECTLY PREDICTED DRAFTED PLAYERS ---")
        for i, player in enumerate(sorted(true_positives, key=lambda x: x['actual_wav'], reverse=True)[:5]):
            draft_model.display_player_prediction(i, player)
    
    # Print a few False Positives (incorrectly predicted to be drafted)
    if false_positives:
        print("\n--- PLAYERS INCORRECTLY PREDICTED TO BE DRAFTED ---")
        for i, player in enumerate(sorted(false_positives, key=lambda x: x['pred_draft_prob'], reverse=True)[:3]):
            draft_model.display_player_prediction(i, player)
    
    # Print total counts for each category
    print(f"\nPrediction Summary (Based on Final Models Trained on All Data):")
    print(f"  False Negatives (Drafted but predicted not to be): {len(false_negatives)}")
    print(f"  False Positives (Not drafted but predicted to be): {len(false_positives)}")
    print(f"  True Positives (Correctly predicted as drafted): {len(true_positives)}")
    print(f"  True Negatives (Correctly predicted as not drafted): {len(true_negatives)}")
    print(f"  Total Accuracy: {accuracy:.2%}")
    
    # Save detailed predictions to CSV in the results folder
    os.makedirs(model.predictions_dir, exist_ok=True)
    prediction_df = pd.DataFrame(predictions)
    prediction_df = prediction_df.sort_values(by='pred_wav', ascending=False)
    prediction_path = os.path.join(model.predictions_dir, f"all_predictions_cv.csv")
    prediction_df.to_csv(prediction_path, index=False)

    print("\n=== Model Training Completed ===")
    print(f"CV results and feature importances saved to {model.features_dir}")
    print(f"All predictions saved to {prediction_path}")


# This module is not meant to be run directly
