import os
import pandas as pd
import re
import json
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

def organize_sos_by_team_year():
    """
    Reorganize SOS data into a nested dictionary structure:
    {
        "team": {
            "year": {SOS data}
        }
    }
    """
    # Load SOS data
    sos_file = os.path.join(RAW_DATA_DIR, 'college_sos', 'college_sos_2014_to_2024.csv')
    logger.info(f"Loading SOS data from {sos_file}")
    
    sos_data = pd.read_csv(sos_file)
    logger.info(f"Loaded {len(sos_data)} SOS entries")
    
    # Clean team names
    sos_data['CleanTeam'] = sos_data['Team'].apply(clean_team_name)
    
    # Create the nested dictionary structure
    sos_by_team_year = {}
    
    for _, row in sos_data.iterrows():
        team = row['CleanTeam']
        year = int(row['season_year'])
        
        if team not in sos_by_team_year:
            sos_by_team_year[team] = {}
        
        sos_by_team_year[team][str(year)] = {
            'Rank': int(row['Rank']),
            'Rating': float(row['Rating']),
            'Hi': int(row['Hi']),
            'Lo': int(row['Lo']),
            'Last': int(row['Last'])
        }
    
    # Save the reorganized SOS data
    output_file = os.path.join(PROCESSED_DATA_DIR, 'sos_by_team_year.json')
    with open(output_file, 'w') as f:
        json.dump(sos_by_team_year, f, indent=2)
    
    logger.info(f"Saved reorganized SOS data to {output_file}")
    
    # Print some stats
    teams_count = len(sos_by_team_year)
    total_team_years = sum(len(years) for years in sos_by_team_year.values())
    
    logger.info(f"Statistics:")
    logger.info(f"Number of teams: {teams_count}")
    logger.info(f"Total team-year combinations: {total_team_years}")
    logger.info(f"Average years per team: {total_team_years/teams_count:.2f}")
    
    # Check data for a few example teams
    example_teams = ['Alabama', 'Oregon', 'Ohio State']
    logger.info(f"\nExample data:")
    
    for team in example_teams:
        if team in sos_by_team_year:
            logger.info(f"{team} has data for years: {sorted(sos_by_team_year[team].keys())}")
            # Show sample data for most recent year
            recent_year = max(int(y) for y in sos_by_team_year[team].keys())
            logger.info(f"  {team} {recent_year} SOS data: {sos_by_team_year[team][str(recent_year)]}")
    
    return sos_by_team_year

if __name__ == "__main__":
    organize_sos_by_team_year()
