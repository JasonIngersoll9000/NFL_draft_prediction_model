import os
import json
import pandas as pd
import re
from fuzzywuzzy import process, fuzz
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def clean_team_name(team_name):
    """Remove the record part from team names like 'Auburn (8-4)'"""
    return re.sub(r'\s*\(\d+-\d+(-\d+)?\)\s*', '', team_name).strip()

def normalize_team_name(name):
    """Normalize team name for better matching"""
    # Convert to uppercase for consistent matching
    name = name.upper()
    
    # Common substitutions
    substitutions = {
        'STATE': 'ST',
        'SOUTHERN': 'S',
        'NORTHERN': 'N',
        'EASTERN': 'E',
        'WESTERN': 'W',
        'CENTRAL': 'C',
        'UNIVERSITY': 'UNIV',
        'COLLEGE': 'COL',
        'WASHINGTON': 'WASH',
        'MISSISSIPPI': 'MISS',
        'LOUISIANA': 'LA',
        'CALIFORNIA': 'CAL',
        'BOWLING GREEN': 'BOWL GREEN',
        'NORTHWESTERN': 'NWESTERN',
        'TCU': 'TX CHRISTIAN',
        'TEXAS CHRISTIAN': 'TX CHRISTIAN',
        'OLE MISS': 'MISSISSIPPI',
        'MISS': 'MISSISSIPPI'
    }
    
    for old, new in substitutions.items():
        name = re.sub(rf'\b{old}\b', new, name)
    
    return name

def collect_pff_team_names():
    """Collect all unique team names from PFF data"""
    pff_teams = set()
    
    # Look for all directories that might contain college data
    college_dirs = []
    for root, dirs, files in os.walk(os.path.join(RAW_DATA_DIR, 'pff', 'college')):
        for file in files:
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    if 'team_name' in df.columns:
                        teams = df['team_name'].dropna().unique()
                        for team in teams:
                            if isinstance(team, str):
                                pff_teams.add(team.strip())
                except Exception as e:
                    logger.warning(f"Error reading {file}: {str(e)}")
    
    return sorted(list(pff_teams))

def create_team_mapping():
    """Create a mapping between SOS team names and PFF team names"""
    # Load the SOS data by team and year
    sos_mapping_file = os.path.join(PROCESSED_DATA_DIR, 'sos_by_team_year.json')
    with open(sos_mapping_file, 'r') as f:
        sos_by_team = json.load(f)
    
    # Get all team names from SOS data
    sos_teams = list(sos_by_team.keys())
    logger.info(f"Loaded {len(sos_teams)} teams from SOS data")
    
    # Get all team names from PFF data
    pff_teams = collect_pff_team_names()
    logger.info(f"Found {len(pff_teams)} unique team names in PFF data")
    
    # Create a mapping from PFF team names to clean SOS team names
    team_mapping = {}
    
    # First try direct matching after normalization
    for pff_team in pff_teams:
        normalized_pff = normalize_team_name(pff_team)
        matched = False
        
        for sos_team in sos_teams:
            normalized_sos = normalize_team_name(sos_team)
            
            if normalized_pff == normalized_sos:
                team_mapping[pff_team] = sos_team
                matched = True
                break
        
        if not matched:
            # Use fuzzy matching for teams that couldn't be matched directly
            best_match, score = process.extractOne(normalized_pff, [normalize_team_name(t) for t in sos_teams], scorer=fuzz.token_sort_ratio)
            if score >= 75:  # Threshold for considering it a good match
                best_idx = [normalize_team_name(t) for t in sos_teams].index(best_match)
                team_mapping[pff_team] = sos_teams[best_idx]
                logger.info(f"Fuzzy matched '{pff_team}' to '{sos_teams[best_idx]}' with score {score}")
            else:
                logger.warning(f"Could not find a good match for '{pff_team}' (best: '{best_match}' with score {score})")
    
    # Add manual mappings for teams that couldn't be automatically matched
    # Updated with exact names from SOS data
    manual_mappings = {
        'ARK STATE': 'Arkansas St',
        'FAU': 'Fla Atlantic',      # Updated with exact name
        'FIU': 'Florida Intl',
        'GA SOUTHRN': 'GA Southern', # Updated with exact name
        'LA LAFAYET': 'Louisiana',
        'LA MONROE': 'UL Monroe',
        'SMU': 'S Methodist',
        'UCONN': 'Connecticut',
        'UMASS': 'U Mass',          # Updated with exact name
        'USF': 'S Florida',         # Updated with exact name
        'UTEP': 'TX El Paso',       # Updated with exact name
        'WAKE': 'Wake Forest'
    }
    
    # Add manual mappings to the team_mapping dictionary
    for pff_team, sos_team in manual_mappings.items():
        # Check if the SOS team is in our SOS data
        if sos_team in sos_teams:
            team_mapping[pff_team] = sos_team
            logger.info(f"Added manual mapping '{pff_team}' -> '{sos_team}'")
        else:
            # Try to find a close match in SOS teams
            best_match, score = process.extractOne(sos_team, sos_teams, scorer=fuzz.token_sort_ratio)
            if score >= 85:  # Higher threshold for confidence
                team_mapping[pff_team] = best_match
                logger.info(f"Manual mapping '{pff_team}' -> '{sos_team}' redirected to '{best_match}' with score {score}")
            else:
                logger.warning(f"Manual mapping '{pff_team}' -> '{sos_team}' failed; couldn't find '{sos_team}' in SOS data")
    
    # Save the mapping
    mapping_file = os.path.join(PROCESSED_DATA_DIR, 'pff_to_sos_team_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(team_mapping, f, indent=2, sort_keys=True)
    
    logger.info(f"Created mapping for {len(team_mapping)} teams out of {len(pff_teams)} PFF teams")
    
    # Print some examples
    logger.info("\nSample mappings:")
    sample_count = min(10, len(team_mapping))
    for pff_team, sos_team in list(team_mapping.items())[:sample_count]:
        logger.info(f"  '{pff_team}' -> '{sos_team}'")
    
    # Check for missing teams
    missing = set(pff_teams) - set(team_mapping.keys())
    if missing:
        logger.warning(f"\n{len(missing)} teams could not be matched:")
        for team in sorted(list(missing))[:20]:  # Show first 20 missing teams
            logger.warning(f"  '{team}'")
        if len(missing) > 20:
            logger.warning(f"  ... and {len(missing) - 20} more")
            
    return team_mapping

if __name__ == "__main__":
    create_team_mapping()
