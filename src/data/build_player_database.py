#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to build player database that pulls together data from various sources

Integrates college stats, NFL stats, combine data, draft info, and SOS data.
Created for my NFL draft prediction project - J.I.
"""

import os
import glob
import json
import pickle
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz  # for name matching
import logging
import datetime  # for year calculations

# Set up logging - basic config should be enough for now
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TODO: Maybe add file logging later if needed?

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = PROCESSED_DATA_DIR  # using same dir for now

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_pff_college_data(data_types=None):
    '''Load PFF college files'''
    # Just loads data from the folders - nothing fancy
    if data_types is None:
        data_types = ['rushing grades', 'receiving grades', 'passing grades', 'blocking grades', 'defensive grades']
    
    pff_college_dir = os.path.join(RAW_DATA_DIR, 'pff', 'college')
    data_dict = {}
    
    for data_type in data_types:
        type_dir = os.path.join(pff_college_dir, data_type)
        if not os.path.exists(type_dir):
            logger.warning(f"Directory not found: {type_dir}")
            continue
            
        csv_files = glob.glob(os.path.join(type_dir, '*.csv'))
        if not csv_files:
            logger.warning(f"No CSV files found in {type_dir}")
            continue
            
        # Load and combine all files for this data type
        df_list = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract year from filename
                year = os.path.basename(csv_file).split('_')[-1].split('.')[0]
                df['season_year'] = int(year)
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        if df_list:
            data_dict[data_type.lower().replace(' grades', '')] = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded {len(df_list)} files for {data_type}")
    
    return data_dict

def load_pff_nfl_data(data_types=None):
    # NFL data loading is similar to college data loading
    # but from different folders
    if data_types is None:
        data_types = ['rushing', 'receiving', 'passing', 'blocking', 'defense']
    
    pff_nfl_dir = os.path.join(RAW_DATA_DIR, 'pff', 'NFL')
    data_dict = {}
    
    for data_type in data_types:
        type_dir = os.path.join(pff_nfl_dir, data_type)
        if not os.path.exists(type_dir):
            logger.warning(f"Directory not found: {type_dir}")
            continue
            
        csv_files = glob.glob(os.path.join(type_dir, '*.csv'))
        if not csv_files:
            logger.warning(f"No CSV files found in {type_dir}")
            continue
            
        # Load and combine all files for this data type
        df_list = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract year from filename
                year = os.path.basename(csv_file).split('_')[-1].split('.')[0]
                df['season_year'] = int(year)
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        if df_list:
            data_dict[data_type] = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded {len(df_list)} files for {data_type}")
    
    return data_dict

def load_sos_data():
    """
    Load strength of schedule data.
    
    Returns:
        DataFrame: Combined SOS data with team and year.
    """
    sos_dir = os.path.join(RAW_DATA_DIR, 'college_sos')
    combined_file = os.path.join(sos_dir, 'combined_college_sos.csv')
    
    if os.path.exists(combined_file):
        df = pd.read_csv(combined_file)
        logger.info(f"Loaded combined SOS data with {len(df)} rows")
        return df
    else:
        # Load and combine individual year files
        csv_files = glob.glob(os.path.join(sos_dir, 'college_sos_*.csv'))
        if not csv_files:
            logger.warning(f"No SOS CSV files found in {sos_dir}")
            return pd.DataFrame()
            
        df_list = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract year from filename
                year = os.path.basename(csv_file).split('_')[-1].split('.')[0]
                df['season_year'] = int(year)
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded and combined {len(df_list)} SOS files with {len(combined_df)} rows")
            return combined_df
        else:
            return pd.DataFrame()

def load_draft_data():
    # Grab all the draft csv files and combine them
    draft_dir = os.path.join(RAW_DATA_DIR, 'nfl_draft')
    df_list = []
    
    for year in range(2014, 2024 + 1):
        draft_file = os.path.join(draft_dir, f'draft_{year}.csv')
        if os.path.exists(draft_file):
            try:
                df = pd.read_csv(draft_file)
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {draft_file}: {e}")
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Loaded {len(df_list)} draft files with {len(combined_df)} rows")
        return combined_df
    else:
        return pd.DataFrame()

def load_combine_data():
    # Grab all the combine csv files and combine them
    combine_dir = os.path.join(RAW_DATA_DIR, 'nfl_combine')
    df_list = []
    
    for year in range(2014, 2024 + 1):
        combine_file = os.path.join(combine_dir, f'combine_{year}.csv')
        if os.path.exists(combine_file):
            try:
                df = pd.read_csv(combine_file)
                df_list.append(df)
            except Exception as e:
                logger.error(f"Error loading {combine_file}: {e}")
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Loaded {len(df_list)} combine files with {len(combined_df)} rows")
        return combined_df
    else:
        return pd.DataFrame()

def standardize_names(df, name_col):
    '''
    Clean up player names to make matching easier
    This was a pain to get right - spent way too long on this
    '''
    if name_col not in df.columns:
        logger.warning(f"Column {name_col} not found in DataFrame")
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    result_df = df.copy()
    
    # Standardize names
    result_df['Name_Clean'] = result_df[name_col].str.normalize('NFKD')\
                                              .str.encode('ascii', errors='ignore')\
                                              .str.decode('utf-8')\
                                              .str.lower()\
                                              .str.replace(r'[^\w\s]', '', regex=True)\
                                              .str.replace(r'\s+', ' ', regex=True)\
                                              .str.strip()
    
    logger.info(f"Standardized names in column {name_col}")
    return result_df

def standardize_colleges(df, college_col):
    """
    Standardize college names for consistent matching.
    
    Args:
        df (DataFrame): DataFrame containing college names
        college_col (str): Column name containing college names
        
    Returns:
        DataFrame: DataFrame with standardized college column added
    """
    if college_col not in df.columns:
        logger.warning(f"Column {college_col} not found in DataFrame")
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    result_df = df.copy()
    
    # Load college mapping
    college_mapping_file = os.path.join(PROCESSED_DATA_DIR, 'college_team_mapping.json')
    with open(college_mapping_file, 'r') as f:
        college_mapping = json.load(f)
    
    # Apply standardization
    result_df['College_Clean'] = result_df[college_col].replace(college_mapping)
    
    # Also clean and standardize
    result_df['College_Clean'] = result_df['College_Clean'].str.normalize('NFKD')\
                                                        .str.encode('ascii', errors='ignore')\
                                                        .str.decode('utf-8')\
                                                        .str.lower()\
                                                        .str.replace(r'[^\w\s]', '', regex=True)\
                                                        .str.replace(r'\s+', ' ', regex=True)\
                                                        .str.strip()
    
    logger.info(f"Standardized colleges in column {college_col}")
    return result_df

def load_team_mapping():
    """
    Load the college team name mapping from the JSON file.
    
    Returns:
        dict: Mapping from PFF team names to SOS team names.
    """
    mapping_file = os.path.join(PROCESSED_DATA_DIR, 'pff_to_sos_team_mapping.json')
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            team_mapping = json.load(f)
        logger.info(f"Loaded team mapping with {len(team_mapping)} entries")
        return team_mapping
    else:
        logger.warning(f"Team mapping file not found: {mapping_file}")
        logger.warning("Run create_team_mapping.py first to create this file")
        return {}

def create_player_mapping(pff_data, draft_data, combine_data):
    """Links players across datasets
    
    This was tricky - players have different names in different datasets
    Had to use fuzzy matching to handle name variations
    """
    # Get all unique players from PFF data
    pff_players = []
    for dtype, df in pff_data.items():  # loop through each data type
        if 'player_id' in df.columns and 'player' in df.columns:  # need these cols
            player_subset = df[['player_id', 'player', 'position', 'team_name']].drop_duplicates()
            pff_players.append(player_subset)  # add to our list
    
    if not pff_players:
        logger.warning("No valid player information found in PFF data")
        return {'draft_to_pff': {}, 'combine_to_pff': {}}
        
    master_players = pd.concat(pff_players, ignore_index=True).drop_duplicates()
    
    # Standardize names
    master_players = standardize_names(master_players, 'player')
    
    # Standardize draft data
    if not draft_data.empty and 'Player' in draft_data.columns:
        draft_data = standardize_names(draft_data, 'Player')
        
        if 'College/Univ' in draft_data.columns:
            draft_data = standardize_colleges(draft_data, 'College/Univ')
    
    # Standardize combine data
    if not combine_data.empty and 'Player' in combine_data.columns:
        combine_data = standardize_names(combine_data, 'Player')
        
        if 'College' in combine_data.columns:
            combine_data = standardize_colleges(combine_data, 'College')
    
    # Create mappings
    draft_to_pff = {}
    combine_to_pff = {}
    
    # Try to match draft players to PFF players
    # This is where the magic happens
    if not draft_data.empty:
        num_players = len(draft_data)
        logger.info(f"Finding matches for {num_players} draft players")
        
        # Not the most efficient but it works
        for i, (idx, player) in enumerate(draft_data.iterrows()):
            # Progress update every 500 players
            if i % 500 == 0 and i > 0:
                logger.info(f"Done {i} players so far")
                
            # Skip if no clean name
            if 'Name_Clean' not in player:
                continue
                
            # Find candidates whose name contains this player's name
            # Not perfect but good enough - could use better fuzzy matching later
            matches = master_players[master_players['Name_Clean'].str.contains(player['Name_Clean'], case=False, na=False)]
            
            if len(matches) == 1:
                # Only one match found, use it
                draft_to_pff[idx] = matches.iloc[0]['player_id']
            elif len(matches) > 1:
                # Multiple matches, try to narrow down by position
                if 'Pos' in player and player['Pos'] in matches['position'].values:
                    pos_matches = matches[matches['position'] == player['Pos']]
                    if len(pos_matches) > 0:
                        draft_to_pff[idx] = pos_matches.iloc[0]['player_id']
                    else:
                        draft_to_pff[idx] = matches.iloc[0]['player_id']
                else:
                    draft_to_pff[idx] = matches.iloc[0]['player_id']
    
    # Map combine data to PFF IDs
    if not combine_data.empty:
        total_combine_players = len(combine_data)
        logger.info(f"Mapping {total_combine_players} combine players to PFF IDs")
        
        for i, (idx, combine_player) in enumerate(combine_data.iterrows()):
            if i % 500 == 0 and i > 0:
                logger.info(f"Processed {i}/{total_combine_players} combine players")
                
            if 'Name_Clean' not in combine_player:
                continue
                
            # Use fuzzy matching to find potential matches
            matches = master_players[master_players['Name_Clean'].str.contains(combine_player['Name_Clean'], case=False, na=False)]
            
            if len(matches) == 1:
                # Only one match found, use it
                combine_to_pff[idx] = matches.iloc[0]['player_id']
            elif len(matches) > 1:
                # Multiple matches, try to narrow down by position
                if 'Pos' in combine_player and combine_player['Pos'] in matches['position'].values:
                    pos_matches = matches[matches['position'] == combine_player['Pos']]
                    if len(pos_matches) > 0:
                        combine_to_pff[idx] = pos_matches.iloc[0]['player_id']
                    else:
                        combine_to_pff[idx] = matches.iloc[0]['player_id']
                else:
                    combine_to_pff[idx] = matches.iloc[0]['player_id']
    
    logger.info(f"Mapped {len(draft_to_pff)}/{len(draft_data) if not draft_data.empty else 0} draft players")
    logger.info(f"Mapped {len(combine_to_pff)}/{len(combine_data) if not combine_data.empty else 0} combine players")
    
    return {
        'draft_to_pff': draft_to_pff,
        'combine_to_pff': combine_to_pff
    }

def build_player_database(mapping, pff_college_data, pff_nfl_data, draft_data, combine_data, sos_data, team_mapping=None, sos_by_team_year=None):
    """
    Build comprehensive player database with all data sources.
    
    Args:
        mapping: Dictionary with ID mappings
        pff_college_data: Dictionary of DataFrames with college PFF data
        pff_nfl_data: Dictionary of DataFrames with NFL PFF data
        draft_data: DataFrame with draft information
        combine_data: DataFrame with combine information
        sos_data: DataFrame with strength of schedule data
        team_mapping: Dictionary mapping SOS team names to PFF team names
        sos_by_team_year: Dictionary with SOS data organized by team and year
        
    Returns:
        Dictionary with player database
    """
    player_db = {}
    
    # Get current year for calculating if player might still be in college
    current_year = datetime.datetime.now().year
    
    # Get all unique player IDs from college PFF data
    all_player_ids = set()
    for data_type, df in pff_college_data.items():
        if 'player_id' in df.columns:
            all_player_ids.update(df['player_id'].unique())
    
    logger.info(f"Building database for {len(all_player_ids)} players")
    
    # Convert team mapping to lowercase for consistent matching
    if team_mapping:
        lowercase_team_mapping = {k.lower().strip(): v for k, v in team_mapping.items() if v is not None}
        logger.info(f"Using team mapping with {len(lowercase_team_mapping)} valid entries")
    else:
        lowercase_team_mapping = {}
    
    # Process each player
    for i, player_id in enumerate(all_player_ids):
        # Add progress reporting every 1000 players
        if i > 0 and i % 1000 == 0:
            logger.info(f"Processed {i}/{len(all_player_ids)} players")
            
        player_db[player_id] = {
            'player_info': {},
            'college_stats': {},
            'combine_stats': {},
            'nfl_stats': {},
            'draft_info': {},
            'career_status': 'unknown'  # Default status
        }
        
        # Fill in player info from first available source
        for data_type, df in pff_college_data.items():
            player_matches = df[df['player_id'] == player_id]
            if not player_matches.empty:
                player_db[player_id]['player_info']['name'] = player_matches.iloc[0]['player']
                if 'position' in player_matches.columns:
                    player_db[player_id]['player_info']['position'] = player_matches.iloc[0]['position']
                break
        
        # Fill in college stats
        for data_type, df in pff_college_data.items():
            player_data = df[df['player_id'] == player_id]
            
            for _, row in player_data.iterrows():
                if 'season_year' not in row:
                    continue
                    
                year = row['season_year']
                if year not in player_db[player_id]['college_stats']:
                    player_db[player_id]['college_stats'][year] = {
                        'team': row['team_name'] if 'team_name' in row else None,
                        'rushing': {},
                        'receiving': {},
                        'passing': {},
                        'blocking': {},
                        'defense': {},
                        'sos_data': {}
                    }
                    
                    # Add SOS for this team and year if available
                    if not sos_data.empty and 'team_name' in row and year in sos_data['season_year'].values:
                        # Get the PFF team name
                        pff_team = row['team_name'].strip() if 'team_name' in row else None
                        
                        if pff_team and team_mapping:
                            # Get the corresponding SOS team name using our mapping
                            # Our mapping is now from PFF team -> SOS team
                            sos_team = team_mapping.get(pff_team)
                            
                            if sos_team:
                                sos_matches = sos_data[(sos_data['Team'] == sos_team) & 
                                                 (sos_data['season_year'] == year)]
                                if not sos_matches.empty:
                                    sos_metrics = {}
                                    for col in sos_matches.columns:
                                        if col not in ['Team', 'Rank', 'season_year']:
                                            sos_metrics[col] = sos_matches.iloc[0][col]
                                    player_db[player_id]['college_stats'][year]['sos_data'] = sos_metrics
                    
                    # Add SOS data from sos_by_team_year if available
                    # This is our new, more efficient structure
                    if sos_by_team_year and pff_team and team_mapping:
                        # Get the corresponding SOS team name using our mapping
                        sos_team = team_mapping.get(pff_team)
                        
                        if sos_team and sos_team in sos_by_team_year and str(year) in sos_by_team_year[sos_team]:
                            player_db[player_id]['college_stats'][year]['sos_data'] = sos_by_team_year[sos_team][str(year)]
                            # Add a source indicator
                            player_db[player_id]['college_stats'][year]['sos_data']['source'] = 'organized_sos'
                
                # Add the stats for this data type - optimize dict conversion
                stats_dict = {}
                for col_name in row.index:
                    val = row[col_name]
                    # Convert numpy types to Python native types for better serialization
                    if isinstance(val, (np.integer, np.int64)):
                        stats_dict[col_name] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        stats_dict[col_name] = float(val)
                    elif isinstance(val, np.ndarray):
                        stats_dict[col_name] = val.tolist()
                    elif pd.isna(val):
                        stats_dict[col_name] = None
                    else:
                        stats_dict[col_name] = val
                
                player_db[player_id]['college_stats'][year][data_type] = stats_dict
        
        # Add NFL stats if available
        for data_type, df in pff_nfl_data.items():
            player_data = df[df['player_id'] == player_id] if 'player_id' in df.columns else pd.DataFrame()
            
            for _, row in player_data.iterrows():
                if 'season_year' not in row:
                    continue
                    
                year = row['season_year']
                if year not in player_db[player_id]['nfl_stats']:
                    player_db[player_id]['nfl_stats'][year] = {
                        'team': row['team_name'] if 'team_name' in row else None,
                        'rushing': {},
                        'receiving': {},
                        'passing': {},
                        'blocking': {},
                        'defense': {}
                    }
                
                # Add the stats for this data type - optimize dict conversion
                stats_dict = {}
                for col_name in row.index:
                    val = row[col_name]
                    # Convert numpy types to Python native types for better serialization
                    if isinstance(val, (np.integer, np.int64)):
                        stats_dict[col_name] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        stats_dict[col_name] = float(val)
                    elif isinstance(val, np.ndarray):
                        stats_dict[col_name] = val.tolist()
                    elif pd.isna(val):
                        stats_dict[col_name] = None
                    else:
                        stats_dict[col_name] = val
                
                player_db[player_id]['nfl_stats'][year][data_type] = stats_dict
        
        # Add draft info if this player was drafted
        draft_id = [k for k, v in mapping['draft_to_pff'].items() if v == player_id]
        if draft_id and not draft_data.empty:
            draft_row = draft_data.loc[draft_id[0]]
            
            # Create basic draft info
            player_db[player_id]['draft_info'] = {
                'year': draft_row['draft_year'] if 'draft_year' in draft_row else None,
                'round': draft_row['Rnd'] if 'Rnd' in draft_row else None,
                'pick': draft_row['Pick'] if 'Pick' in draft_row else None,
                'team': draft_row['Tm'] if 'Tm' in draft_row else None
            }
            
            # Add AV metrics and other NFL performance data from draft data
            av_metrics = ['wAV', 'DrAV', 'G', 'AP1', 'PB', 'St']
            for metric in av_metrics:
                if metric in draft_row and pd.notnull(draft_row[metric]):
                    # Convert to the appropriate numeric type
                    try:
                        if metric == 'G' or metric == 'AP1' or metric == 'PB' or metric == 'St':
                            # These should be integers
                            player_db[player_id]['draft_info'][metric] = int(draft_row[metric])
                        else:
                            # wAV and DrAV should be floats
                            player_db[player_id]['draft_info'][metric] = float(draft_row[metric])
                    except (ValueError, TypeError):
                        # If conversion fails, store as the original type
                        player_db[player_id]['draft_info'][metric] = draft_row[metric]
    
            # Calculate AV per game if games played data is available
            try:
                if 'wAV' in player_db[player_id]['draft_info'] and 'G' in player_db[player_id]['draft_info']:
                    wAV = player_db[player_id]['draft_info']['wAV']
                    games = player_db[player_id]['draft_info']['G']
                    
                    if isinstance(games, (int, float)) and games > 0 and isinstance(wAV, (int, float)):
                        player_db[player_id]['draft_info']['av_per_game'] = float(wAV) / float(games)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                # Skip calculation if there's an error
                logger.warning(f"Error calculating AV per game for player {player_id}: {e}")
        
        # Add combine data if available
        combine_id = [k for k, v in mapping['combine_to_pff'].items() if v == player_id]
        if combine_id and not combine_data.empty:
            combine_row = combine_data.loc[combine_id[0]]
            player_db[player_id]['combine_stats'] = {k: v for k, v in combine_row.to_dict().items() 
                                                 if k not in ['Name', 'Name_Clean', 'College', 'College_Clean']}
        
        # Figure out where the player is in their career path
        # This turned out to be trickier than I thought
        has_nfl = bool(player_db[player_id]['nfl_stats'])
        was_drafted = bool(player_db[player_id]['draft_info'])
        
        # When was their last college season?
        last_yr = 0
        if player_db[player_id]['college_stats']:
            # Just grab the years and find the most recent one
            years = [int(y) for y in player_db[player_id]['college_stats'].keys()]
            if years:  # make sure the list isn't empty
                last_yr = max(years)
        
        # Set their status - accuracy could be improved later
        if has_nfl:
            # Made it to the NFL
            player_db[player_id]['career_status'] = 'nfl'
        elif was_drafted and not has_nfl:
            # Got drafted but maybe no stats yet
            # or maybe a bust who didn't make final roster
            player_db[player_id]['career_status'] = 'drafted_no_nfl'
        elif last_yr >= current_year - 1:  
            # Still in college probably
            player_db[player_id]['career_status'] = 'active_college'
        elif last_yr > 0:
            # Done with college but didn't make NFL
            player_db[player_id]['career_status'] = 'completed_no_nfl'
        else:
            # Couldn't determine status
            player_db[player_id]['career_status'] = 'unknown'
    
    logger.info(f"Built player database with {len(player_db)} player entries")
    return player_db

def load_sos_by_team_year():
    '''Gets the pre-processed SOS data by team/year'''
    # Saved this to avoid reprocessing every time - much faster
    sos_by_team_year_file = os.path.join(PROCESSED_DATA_DIR, 'sos_by_team_year.json')
    
    if os.path.exists(sos_by_team_year_file):
        with open(sos_by_team_year_file, 'r') as f:
            sos_by_team_year = json.load(f)
        logger.info(f"Loaded SOS data by team/year with {len(sos_by_team_year)} entries")
        return sos_by_team_year
    else:
        logger.warning(f"SOS data by team/year file not found: {sos_by_team_year_file}")
        logger.warning("Run prepare_sos_by_team.py first to create this file")
        return {}

def prepare_sos_by_team(sos_data):
    # Organize SOS data by team and year for easier lookup
    # TODO: Could make this more efficient but works for now
    sos_by_team_year = {}
    
    for _, row in sos_data.iterrows():
        team = row['Team']
        year = str(row['season_year'])
        
        if team not in sos_by_team_year:
            sos_by_team_year[team] = {}
        
        if year not in sos_by_team_year[team]:
            sos_by_team_year[team][year] = {}
        
        for col in row.index:
            if col not in ['Team', 'Rank', 'season_year']:
                sos_by_team_year[team][year][col] = row[col]
    
    return sos_by_team_year

# Helper function for JSON stuff
def json_serializable(obj):
    # Convert numpy stuff to regular Python types so JSON doesn't break
    # Stupid numpy types aren't JSON serializable
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
        return obj.isoformat()
    else:
        # Just convert to string as fallback
        return str(obj)

def main():
    # Main script function - runs everything
    # Could break this into smaller funcs but this works fine
    logger.info("Starting to build player database")
    
    # Load PFF college data
    logger.info("Loading PFF college data")
    pff_college_data = load_pff_college_data()
    
    # Load PFF NFL data
    logger.info("Loading PFF NFL data")
    pff_nfl_data = load_pff_nfl_data()
    
    # Load SOS data (both formats for backward compatibility)
    logger.info("Loading SOS data")
    sos_data = load_sos_data()
    
    # Load reorganized SOS data by team and year
    logger.info("Loading reorganized SOS data by team and year")
    sos_by_team_year = load_sos_by_team_year()
    
    if not sos_by_team_year:
        # Create it if it doesn't exist
        sos_by_team_year = prepare_sos_by_team(sos_data)
        
        # Save it for next time
        sos_by_team_year_file = os.path.join(PROCESSED_DATA_DIR, 'sos_by_team_year.json')
        with open(sos_by_team_year_file, 'w') as f:
            json.dump(sos_by_team_year, f)
        logger.info(f"Saved SOS data by team/year to {sos_by_team_year_file}")
    
    # Load draft data
    logger.info("Loading draft data")
    draft_data = load_draft_data()
    
    # Load combine data
    logger.info("Loading combine data")
    combine_data = load_combine_data()
    
    # Create player ID mapping
    logger.info("Creating player ID mapping")
    mapping = create_player_mapping(pff_college_data, draft_data, combine_data)
    
    # Load team mapping
    logger.info("Loading team mapping")
    team_mapping = load_team_mapping()
    
    # Build the player database
    logger.info("Building player database")
    player_db = build_player_database(mapping, pff_college_data, pff_nfl_data, 
                                     draft_data, combine_data, sos_data, team_mapping, sos_by_team_year)
    
    # Save the player database
    db_file = os.path.join(PROCESSED_DATA_DIR, 'player_database.pkl')
    with open(db_file, 'wb') as f:
        pickle.dump(player_db, f)
    logger.info(f"Saved player database to {db_file}")
    
    # Save a JSON version for easy inspection
    json_file = os.path.join(PROCESSED_DATA_DIR, 'player_database_sample.json')
    
    # Converting to JSON in batches to avoid memory issues
    logger.info("Converting sample to JSON (this may take a while)...")
    
    # Get a sample of players to save (first 100 to keep file size manageable)
    sample_players = list(player_db.keys())[:100]
    
    # Create a new dictionary with string keys
    sample_db = {}

    try:
        for player_id in sample_players:
            # Convert player_id to a regular Python int
            str_player_id = str(player_id)
            
            # Deep copy and convert the player data
            player_data = player_db[player_id]
            sample_db[str_player_id] = player_data
            
        with open(json_file, 'w') as f:
            json.dump(sample_db, f, indent=2, default=json_serializable)
        logger.info(f"Saved sample of 100 players to {json_file} for inspection")
    except Exception as e:
        logger.error(f"Error saving JSON sample: {e}")
        logger.info("Continuing without JSON sample")
    
    # Print some stats about what we built
    # Quick summary for my reference
    c_years = set()
    n_years = set()
    
    for p_data in player_db.values():
        c_years.update(p_data['college_stats'].keys())
        n_years.update(p_data['nfl_stats'].keys())
    
    # This helps me check everything worked
    logger.info(f"DB Stats:")
    logger.info(f"Players: {len(player_db)}")
    logger.info(f"College yrs: {sorted(c_years)}")
    logger.info(f"NFL yrs: {sorted(n_years)}")
    
    # Count special data types
    draft_count = sum(1 for p in player_db.values() if p['draft_info'])
    combine_count = sum(1 for p in player_db.values() if p['combine_stats'])
    sos_count = sum(1 for p in player_db.values() if any('sos_data' in y for y in p['college_stats'].values()))
    
    logger.info(f"Draft: {draft_count}, Combine: {combine_count}, SOS: {sos_count}")
    
    # FIXME: Should probably add more validation checks?
    
    return player_db

if __name__ == "__main__":
    main()
