"""
Utility functions for data collection and processing.

This module provides common functionality for the data collection scripts,
including web scraping utilities, data storage helpers, and logging setup.
"""

import os
import time
import random
import requests
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Union, List, Optional

# Project directory paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(LOGS_DIR / f"{name}.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def make_request(url: str, 
                 headers: Optional[Dict[str, str]] = None, 
                 params: Optional[Dict[str, Any]] = None,
                 rate_limit_delay: float = 2.0) -> requests.Response:
    """
    Make an HTTP request with rate limiting and error handling.
    
    Args:
        url: URL to request
        headers: Optional request headers
        params: Optional request parameters
        rate_limit_delay: Seconds to wait between requests (default: 2)
        
    Returns:
        requests.Response: Response object
        
    Raises:
        requests.RequestException: If the request fails
    """
    # Default headers if none provided
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    # Add jitter to rate limit delay
    delay = rate_limit_delay + random.uniform(0, 1)
    time.sleep(delay)
    
    # Make the request
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    return response

def save_dataframe(df: pd.DataFrame, 
                  category: str, 
                  filename: str, 
                  data_type: str = 'raw') -> str:
    """
    Save a pandas DataFrame to the appropriate data directory.
    
    Args:
        df: DataFrame to save
        category: Data category (e.g., 'college_stats', 'nfl_combine')
        filename: Name of the file (without extension)
        data_type: Type of data ('raw', 'interim', or 'processed')
        
    Returns:
        str: Path to the saved file
    """
    if data_type == 'raw':
        dir_path = RAW_DATA_DIR / category
    elif data_type == 'interim':
        dir_path = INTERIM_DATA_DIR / category
    elif data_type == 'processed':
        dir_path = PROCESSED_DATA_DIR / category
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    dir_path.mkdir(exist_ok=True)
    filepath = dir_path / f"{filename}.csv"
    
    df.to_csv(filepath, index=False)
    return str(filepath)

def load_dataframe(category: str, 
                  filename: str, 
                  data_type: str = 'raw') -> pd.DataFrame:
    """
    Load a pandas DataFrame from the appropriate data directory.
    
    Args:
        category: Data category (e.g., 'college_stats', 'nfl_combine')
        filename: Name of the file (without extension)
        data_type: Type of data ('raw', 'interim', or 'processed')
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if data_type == 'raw':
        dir_path = RAW_DATA_DIR / category
    elif data_type == 'interim':
        dir_path = INTERIM_DATA_DIR / category
    elif data_type == 'processed':
        dir_path = PROCESSED_DATA_DIR / category
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    filepath = dir_path / f"{filename}.csv"
    
    return pd.read_csv(filepath)

def get_data_files(category: str, data_type: str = 'raw') -> List[str]:
    """
    Get a list of available data files for a category.
    
    Args:
        category: Data category (e.g., 'college_stats', 'nfl_combine')
        data_type: Type of data ('raw', 'interim', or 'processed')
        
    Returns:
        List[str]: List of filenames without extensions
    """
    if data_type == 'raw':
        dir_path = RAW_DATA_DIR / category
    elif data_type == 'interim':
        dir_path = INTERIM_DATA_DIR / category
    elif data_type == 'processed':
        dir_path = PROCESSED_DATA_DIR / category
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    if not dir_path.exists():
        return []
    
    return [f.stem for f in dir_path.glob('*.csv')]

# NFL position groupings
NFL_POSITION_GROUPS = {
    'QB': ['QB'],
    'RB': ['RB', 'FB', 'HB'],
    'WR': ['WR'],
    'TE': ['TE'],
    'OL': ['C', 'G', 'OG', 'OT', 'T'],
    'DL': ['DE', 'DT', 'NT'],
    'EDGE': ['OLB', 'DE'],  # Edge rushers can be OLB or DE
    'LB': ['ILB', 'MLB', 'LB'],
    'CB': ['CB'],
    'S': ['S', 'FS', 'SS'],
    'ST': ['K', 'P', 'LS']  # Special teams
}
