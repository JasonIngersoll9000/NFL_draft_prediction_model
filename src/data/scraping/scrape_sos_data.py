import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

def create_directory_if_not_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def get_sos_data(year):
    """
    Scrape strength of schedule data for a specific year from TeamRankings.com
    
    Args:
        year (int): The year to scrape data for (e.g., 2023)
        
    Returns:
        pandas.DataFrame: DataFrame containing the scraped SOS data
    """
    # Convert year to a date in December when season rankings are mostly complete
    date_str = f"{year}-12-01"
    
    # Create URL with the date parameter to get historical data
    url = f"https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other?date={date_str}"
    
    print(f"Scraping SOS data for year {year}...")
    
    try:
        # Send request with headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table containing the strength of schedule rankings
        table = soup.find('table', class_='tr-table datatable scrollable')
        
        if table is None:
            print(f"No SOS table found for year {year}")
            return None
        
        # Extract the data
        headers = []
        header_row = table.find('tr', class_='headers')
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all('th')]
        
        if not headers:
            # Try alternative header extraction if the above fails
            headers = [th.text.strip() for th in table.find_all('th')]
        
        # Extract rows
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header row
            row = [td.text.strip() for td in tr.find_all('td')]
            if row and len(row) == len(headers):
                rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Add year column
        df['season_year'] = year
        
        return df
    
    except Exception as e:
        print(f"Error scraping SOS data for year {year}: {e}")
        return None

def scrape_sos_data(start_year, end_year=None):
    """
    Scrape strength of schedule data for a range of years and save to CSV
    
    Args:
        start_year (int): The first year to scrape data for
        end_year (int, optional): The last year to scrape data for. If None, only scrape start_year.
    """
    # Set up the output directory
    output_dir = os.path.join(os.getcwd(), "data", "raw", "college_sos")
    create_directory_if_not_exists(output_dir)
    
    # Determine years to scrape
    years_to_scrape = [start_year] if end_year is None else range(start_year, end_year + 1)
    
    all_data = []
    
    for year in years_to_scrape:
        df = get_sos_data(year)
        
        if df is not None and not df.empty:
            # Save year-specific data
            output_file = os.path.join(output_dir, f"college_sos_{year}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved SOS data for {year} to {output_file}")
            
            all_data.append(df)
        
        # Be respectful with rate limiting
        if year != years_to_scrape[-1]:
            print("Waiting before next request...")
            time.sleep(2)
    
    # Combine all years into a single file if multiple years were scraped
    if len(all_data) > 1:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_output_file = os.path.join(output_dir, f"college_sos_{start_year}_to_{end_year}.csv")
        combined_df.to_csv(combined_output_file, index=False)
        print(f"Saved combined SOS data to {combined_output_file}")

if __name__ == "__main__":
    print("College Football Strength of Schedule Data Scraper")
    print("=================================================")
    
    # Define the range of years to scrape (modify as needed)
    start_year = 2014  # Adjust based on your needs
    end_year = 2024    # Adjust based on your needs
    
    scrape_sos_data(start_year, end_year)
    
    print("\nScraping completed!")
