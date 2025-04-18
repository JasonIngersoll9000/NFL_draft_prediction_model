"""
Simple NFL Combine Scraper

Gets combine data from Pro Football Reference for years 2014-2024.

Usage:
    python scrape_nfl_combine.py --year 2024
    python scrape_nfl_combine.py --years 2022 2023 2024
    python scrape_nfl_combine.py --college_year 2023  # gets combine year 2024
"""

import os
import time
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Save directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                          'data', 'raw', 'nfl_combine')
os.makedirs(DATA_DIR, exist_ok=True)

def get_combine_data(year):
    """Get combine data for a specific year"""
    url = f"https://www.pro-football-reference.com/draft/{year}-combine.htm"
    
    logger.info(f"Getting combine data from {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        # Wait a bit to be nice to the server
        time.sleep(2)
        
        # Get the page
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Failed to get data: status code {response.status_code}")
            return pd.DataFrame()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table - less complex approach
        tables = soup.find_all('table')
        if not tables:
            logger.warning(f"No tables found for {year}")
            return pd.DataFrame()
            
        # Assuming the first table is what we want
        table = tables[0]
        
        # Convert to DataFrame
        df = pd.read_html(str(table))[0]
        
        # Add metadata
        df['combine_year'] = year
        df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
        
    except Exception as e:
        logger.warning(f"Error getting data for {year}: {e}")
        return pd.DataFrame()

def save_data(df, year):
    """Save data to CSV file"""
    if df.empty:
        logger.warning(f"No data to save for {year}")
        return
    
    filename = os.path.join(DATA_DIR, f"combine_{year}.csv")
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(df)} records to {filename}")
    
    # Print some info about what we got
    print(f"\nSample of combine data for {year}:")
    print(df.head(3))
    print(f"\nGot {len(df)} total players")
    print(f"Columns: {', '.join(df.columns)}")

def main():
    parser = argparse.ArgumentParser(description='Scrape NFL combine data')
    parser.add_argument('--year', type=int, help='Combine year (e.g., 2024)')
    parser.add_argument('--years', type=int, nargs='+', help='Multiple years (e.g., 2022 2023 2024)')
    parser.add_argument('--college_year', type=int, help='College year - gets the next year combine data')
    
    args = parser.parse_args()
    
    years = []
    
    # Get years to process
    if args.year:
        years.append(args.year)
    if args.years:
        years.extend(args.years)
    if args.college_year:
        # Next year's combine after college season
        combine_year = args.college_year + 1
        years.append(combine_year)
        print(f"College year {args.college_year} -> Combine year {combine_year}")
    
    if not years:
        print("Error: Specify at least one year with --year, --years, or --college_year")
        return
    
    # Remove duplicates and sort
    years = sorted(list(set(years)))
    print(f"Getting combine data for years: {years}")
    
    # Process each year
    for year in years:
        print(f"\nProcessing year {year}...")
        df = get_combine_data(year)
        
        if not df.empty:
            save_data(df, year)
        else:
            print(f"No data found for {year}")

if __name__ == "__main__":
    main()
