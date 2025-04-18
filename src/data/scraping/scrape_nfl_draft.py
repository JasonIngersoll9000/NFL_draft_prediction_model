#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Define directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'nfl_draft')

# Create directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def get_draft_data(year):
    """
    Get NFL draft data for a specific year from Pro Football Reference
    
    Args:
        year (int): The draft year to scrape
        
    Returns:
        pandas.DataFrame: DataFrame containing the draft data
    """
    url = f"https://www.pro-football-reference.com/years/{year}/draft.htm"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    logging.info(f"Getting draft data from {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        logging.error(f"Failed to get data from {url}, status code: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main draft table
    table = soup.find('table', id='drafts')
    
    if not table:
        logging.error(f"Could not find draft table for year {year}")
        return None
    
    # Read the table using pandas
    df = pd.read_html(str(table))[0]
    
    # Clean the dataframe
    # First, if there are multiple header rows, keep only the last one
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(-1)
    
    # Drop any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Add metadata columns
    df['draft_year'] = year
    df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return df

def save_data(df, year):
    """
    Save the draft data to a CSV file
    
    Args:
        df (pandas.DataFrame): DataFrame containing the draft data
        year (int): The draft year
        
    Returns:
        bool: True if successful, False otherwise
    """
    if df is None or df.empty:
        logging.error(f"No data to save for year {year}")
        return False
    
    filename = os.path.join(DATA_DIR, f"draft_{year}.csv")
    df.to_csv(filename, index=False)
    logging.info(f"Saved {len(df)} records to {filename}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Scrape NFL draft data from Pro Football Reference')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--year', type=int, help='Draft year to scrape')
    group.add_argument('--years', type=int, nargs='+', help='Multiple draft years to scrape')
    
    args = parser.parse_args()
    
    if args.year:
        years = [args.year]
    else:
        years = args.years
    
    print(f"Getting draft data for years: {years}\n")
    
    for year in years:
        print(f"Processing year {year}...\n")
        df = get_draft_data(year)
        
        if df is not None:
            # Display some sample data
            print(f"Sample of draft data for {year}:")
            print(df.head(3))
            print()
            
            # Show the total number of players and column names
            print(f"Got {len(df)} total draft picks")
            print(f"Columns: {', '.join(df.columns)}")
            print()
            
            save_data(df, year)

if __name__ == "__main__":
    main()
