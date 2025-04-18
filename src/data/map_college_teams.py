#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a mapping between college team names in the SOS data and PFF data.
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_pff_team_names():
    """
    Load a sample PFF file and extract unique team names.
    
    Returns:
        set: Set of unique team names from PFF data
    """
    pff_college_dir = os.path.join(RAW_DATA_DIR, 'pff', 'college')
    
    # Just need to look at one file to get team names
    sample_subdirs = ['Rushing Grades', 'receiving grades', 'passing grades', 'blocking grades', 'defensive grades']
    all_teams = set()
    
    for subdir in sample_subdirs:
        dir_path = os.path.join(pff_college_dir, subdir)
        if not os.path.exists(dir_path):
            continue
            
        # Get the first CSV file in the directory
        csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
        if not csv_files:
            continue
            
        try:
            # Load the first file
            df = pd.read_csv(csv_files[0])
            if 'team_name' in df.columns:
                # Add unique team names to the set
                teams = df['team_name'].dropna().unique()
                all_teams.update(teams)
                logger.info(f"Found {len(teams)} unique teams in {os.path.basename(csv_files[0])}")
                # Once we have teams, we can break
                if all_teams:
                    break
        except Exception as e:
            logger.error(f"Error processing {csv_files[0]}: {e}")
    
    logger.info(f"Extracted {len(all_teams)} unique team names from PFF college data")
    return all_teams

def load_sos_data():
    """
    Load college strength of schedule data.
    
    Returns:
        DataFrame: Combined strength of schedule data across years.
    """
    sos_dir = os.path.join(RAW_DATA_DIR, 'college_sos')
    
    if not os.path.exists(sos_dir):
        logger.warning(f"SOS directory not found: {sos_dir}")
        return pd.DataFrame()
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(sos_dir, '*.csv'))
    
    # Load and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(dfs)} SOS files with {len(combined_df)} total rows")
        return combined_df
    else:
        logger.warning("No SOS data loaded")
        return pd.DataFrame()

def extract_clean_sos_team_names(sos_data):
    """
    Extract unique team names from SOS data and clean them by removing record information.
    
    Args:
        sos_data (DataFrame): The SOS data
        
    Returns:
        dict: Dictionary mapping original team names to cleaned names
    """
    sos_teams_mapping = {}
    
    if not sos_data.empty and 'Team' in sos_data.columns:
        for team in sos_data['Team'].dropna().unique():
            # Remove the record part (e.g., " (2-10)")
            cleaned_team = re.sub(r'\s*\(\d+-\d+\)$', '', team)
            sos_teams_mapping[team] = cleaned_team
    
    logger.info(f"Extracted {len(sos_teams_mapping)} unique team names from SOS data")
    return sos_teams_mapping

def create_team_name_mapping(pff_teams, sos_teams_mapping):
    """
    Create a mapping between SOS and PFF team names using fuzzy matching.
    
    Args:
        pff_teams (set): Set of team names from PFF data
        sos_teams_mapping (dict): Dict mapping original SOS team names to cleaned names
    
    Returns:
        dict: Mapping from SOS team names to PFF team names
    """
    # Convert to list for fuzzy matching
    pff_teams_list = list(pff_teams)
    
    # Create a lowercase version of PFF teams for case-insensitive matching
    pff_teams_lower = {team.lower(): team for team in pff_teams}
    
    # Create a set of manual mappings for common naming discrepancies
    manual_mappings = {
        "Alabama": "ALABAMA",
        "App State": "APP STATE",
        "Arizona St": "ARIZONA ST",
        "Arkansas St": "ARK STATE",
        "Army": "ARMY",
        "Auburn": "AUBURN",
        "Ball St": "BALL STATE",
        "Baylor": "BAYLOR",
        "Boise St": "BOISE ST", 
        "Boston Col": "BOSTON COLLEGE",
        "Bowling Grn": "BOWL GREEN",
        "Buffalo": "BUFFALO",
        "BYU": "BYU",
        "California": "CAL",
        "Central Mich": "CENTRAL MICHIGAN",
        "Charlotte": "CHARLOTTE",
        "Cincinnati": "CINCINNATI",
        "Clemson": "CLEMSON",
        "Coastal Car": "COASTAL CAROLINA",
        "Colorado St": "COLORADO ST",
        "Connecticut": "UCONN",
        "Duke": "DUKE",
        "E Carolina": "E CAROLINA",
        "E Michigan": "E MICHIGAN",
        "Fla Atlantic": "FLORIDA ATL",
        "Florida Intl": "FIU",
        "Florida St": "FLORIDA ST",
        "Fresno St": "FRESNO ST",
        "GA Southern": "GA SOUTHERN",
        "Georgia St": "GEORGIA STATE",
        "GA Tech": "GA TECH",
        "Hawaii": "HAWAII",
        "Illinois": "ILLINOIS",
        "Indiana": "INDIANA",
        "Iowa": "IOWA",
        "Iowa St": "IOWA ST",
        "James Mad": "JAMES MADISON",
        "Jksnville St": "JACKSONVILLE ST",
        "Kansas": "KANSAS",
        "Kansas St": "KANSAS ST",
        "Kent St": "KENT STATE",
        "Kennesaw St": "KENNESAW ST",
        "Kentucky": "KENTUCKY",
        "LA Lafayette": "LOUISIANA",
        "Louisiana": "LOUISIANA",
        "Liberty": "LIBERTY",
        "UL Monroe": "UL MONROE",
        "LA Monroe": "UL MONROE",
        "LA Tech": "LA TECH",
        "Louisville": "LOUISVILLE",
        "LSU": "LSU",
        "Marshall": "MARSHALL",
        "Maryland": "MARYLAND",
        "Memphis": "MEMPHIS",
        "Miami (FL)": "MIAMI FL",
        "Miami": "MIAMI FL",
        "Miami (OH)": "MIAMI OH",
        "Michigan": "MICHIGAN",
        "Michigan St": "MICHIGAN ST",
        "Middle Tenn": "MIDDLE TENN",
        "Minnesota": "MINNESOTA",
        "Mississippi": "OLE MISS",
        "Miss State": "MISSISSIPPI ST",
        "Missouri": "MISSOURI",
        "N Carolina": "NORTH CAROLINA",
        "N Illinois": "N ILLINOIS",
        "N Mex State": "NEW MEXICO ST",
        "Navy": "NAVY",
        "NC State": "NC STATE",
        "Nebraska": "NEBRASKA",
        "Nevada": "NEVADA",
        "New Mexico": "NEW MEXICO",
        "North Texas": "NORTH TEXAS",
        "Northwestern": "NORTHWESTERN",
        "Notre Dame": "NOTRE DAME",
        "Ohio": "OHIO",
        "Ohio St": "OHIO STATE",
        "Oklahoma": "OKLAHOMA",
        "Oklahoma St": "OKLAHOMA ST",
        "Old Dominion": "DOMINION",
        "Oregon": "OREGON",
        "Oregon St": "OREGON ST",
        "Penn St": "PENN ST",
        "Pittsburgh": "PITTSBURGH",
        "Purdue": "PURDUE",
        "Rice": "RICE",
        "Rutgers": "RUTGERS",
        "S Alabama": "S ALABAMA",
        "S Carolina": "S CAROLINA",
        "S Florida": "S FLORIDA",
        "S Methodist": "SMU",
        "S Mississippi": "SOUTHERN MISS",
        "Sam Hous St": "SAM HOUSTON ST",
        "San Diego St": "SAN DIEGO ST",
        "San Jose St": "S JOSE ST",
        "Stanford": "STANFORD",
        "Syracuse": "SYRACUSE",
        "TCU": "TCU",
        "TX Christian": "TCU",
        "Temple": "TEMPLE",
        "Tennessee": "TENNESSEE",
        "Texas": "TEXAS",
        "Texas A&M": "TEXAS A&M",
        "Texas St": "TEXAS ST",
        "Texas Tech": "TEXAS TECH",
        "Toledo": "TOLEDO",
        "Troy": "TROY",
        "Tulane": "TULANE",
        "Tulsa": "TULSA",
        "U Mass": "MASSACHUSETTS",
        "UAB": "UAB",
        "UCF": "UCF",
        "UCLA": "UCLA",
        "UNLV": "UNLV",
        "USC": "USC",
        "Utah": "UTAH",
        "Utah St": "UTAH ST",
        "UTEP": "UTEP",
        "TX El Paso": "UTEP",
        "UTSA": "UTSA",
        "Vanderbilt": "VANDERBILT",
        "Virginia": "VIRGINIA",
        "VA Tech": "VIRGINIA TECH",
        "Wake Forest": "WAKE FOREST",
        "Washington": "WASHINGTON",
        "Washington St": "WASH STATE",
        "Wash State": "WASH STATE", 
        "W Kentucky": "W KENTUCKY",
        "W Michigan": "W MICHIGAN",
        "W Virginia": "W VIRGINIA",
        "Wisconsin": "WISCONSIN",
        "Wyoming": "WYOMING"
    }
    
    # Create mapping from cleaned SOS team names to PFF team names
    mapping = {}
    
    # Apply manual mappings first
    for original, cleaned in sos_teams_mapping.items():
        if cleaned in manual_mappings:
            manual_match = manual_mappings[cleaned]
            
            # Try exact match first (case-insensitive)
            if manual_match.upper() in pff_teams:
                mapping[original] = manual_match.upper()
                continue
                
            if manual_match.lower() in pff_teams_lower:
                mapping[original] = pff_teams_lower[manual_match.lower()]
                continue
                
            # Try a fuzzy match against PFF teams for our manual mapping
            best_match = process.extractOne(manual_match, pff_teams_list, 
                                          scorer=fuzz.token_sort_ratio,
                                          score_cutoff=80)
            if best_match:
                mapping[original] = best_match[0]
                continue
    
    # For teams without a successful mapping yet, try direct fuzzy matching
    for original, cleaned in sos_teams_mapping.items():
        if original in mapping:
            continue  # Skip if already mapped
            
        # Try to find best match in PFF teams
        best_match = process.extractOne(cleaned, pff_teams_list, 
                                      scorer=fuzz.token_sort_ratio,
                                      score_cutoff=75)  # Lower threshold slightly
        if best_match:
            mapping[original] = best_match[0]
        else:
            # No good match found, log and add to unmatched list
            logger.warning(f"No good match found for SOS team: {cleaned}")
            mapping[original] = None
    
    # Count successful mappings
    successful = sum(1 for v in mapping.values() if v is not None)
    logger.info(f"Created mappings for {successful}/{len(mapping)} SOS team names")
    
    return mapping

def main():
    """Main function to extract and map college team names."""
    logger.info("Starting college team name mapping extraction")
    
    # Load PFF team names
    pff_teams = load_pff_team_names()
    
    # Print some PFF team names for inspection
    print("\nSample PFF team names:")
    sample_size = min(20, len(pff_teams))
    for team in list(pff_teams)[:sample_size]:
        print(f"  - \"{team}\"")
    
    # Load SOS data
    sos_data = load_sos_data()
    
    # Extract and clean SOS team names
    sos_teams_mapping = extract_clean_sos_team_names(sos_data)
    
    # Print some SOS team names for inspection
    print("\nSample SOS team names (original -> cleaned):")
    sample_size = min(20, len(sos_teams_mapping))
    sample_items = list(sos_teams_mapping.items())[:sample_size]
    for orig, cleaned in sample_items:
        print(f"  - \"{orig}\" -> \"{cleaned}\"")
    
    # Create mapping between SOS and PFF team names
    team_mapping = create_team_name_mapping(pff_teams, sos_teams_mapping)
    
    # Save the team mapping
    mapping_file = os.path.join(PROCESSED_DATA_DIR, 'college_team_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(team_mapping, f, indent=2)
    logger.info(f"Saved team mapping to {mapping_file}")
    
    # Print statistics
    print(f"\nTotal PFF teams: {len(pff_teams)}")
    print(f"Total SOS teams: {len(sos_teams_mapping)}")
    print(f"Successfully mapped: {sum(1 for v in team_mapping.values() if v is not None)}")
    print(f"Unmapped: {sum(1 for v in team_mapping.values() if v is None)}")
    
    # Print sample of successful mappings
    successful = {k: v for k, v in team_mapping.items() if v is not None}
    if successful:
        print("\nSample successful mappings (SOS original -> PFF team):")
        sample_size = min(20, len(successful))
        sample_items = list(successful.items())[:sample_size]
        for sos_orig, pff_team in sample_items:
            print(f"  - \"{sos_orig}\" -> \"{pff_team}\"")
    
    # Print unmapped teams for manual review
    unmapped = {k: sos_teams_mapping[k] for k in team_mapping if team_mapping[k] is None}
    if unmapped:
        print("\nUnmapped teams (SOS cleaned name):")
        for orig, cleaned in sorted(unmapped.items(), key=lambda x: x[1]):
            print(f"  - \"{cleaned}\": \"\",  # Original: {orig}")
    
    return team_mapping

if __name__ == "__main__":
    main()
