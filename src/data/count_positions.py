#!/usr/bin/env python
"""
Script to count the number of players by position in the player database.
"""

import json
import pickle
from collections import Counter
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

def count_positions():
    """
    Count the number of players by position in the player database.
    
    Returns:
        tuple: A tuple containing (position_counts, total_count)
    """
    # Path to the player database
    db_path = project_root / "data" / "processed" / "player_database.pkl"
    
    print(f"Loading player database from {db_path}...")
    
    try:
        # Load the database
        with open(db_path, 'rb') as f:
            player_db = pickle.load(f)
        
        # Count positions
        positions = []
        for player_id, player_data in player_db.items():
            if "player_info" in player_data and "position" in player_data["player_info"]:
                positions.append(player_data["player_info"]["position"])
        
        # Count occurrences of each position
        position_counts = Counter(positions)
        
        return position_counts, len(positions)
    
    except FileNotFoundError:
        print(f"Error: Could not find player database at {db_path}")
        return Counter(), 0
    except Exception as e:
        print(f"Error: {e}")
        return Counter(), 0

def main():
    """Main function to run the script."""
    position_counts, total_count = count_positions()
    
    if total_count == 0:
        print("No players found in the database.")
        return
    
    # Print results
    print(f"\nTotal players in database: {total_count}\n")
    print("Position counts:")
    print("-" * 30)
    
    # Sort by count (descending)
    for position, count in position_counts.most_common():
        percentage = (count / total_count) * 100
        print(f"{position:<5}: {count:>5} ({percentage:.2f}%)")
    
    # Save results to a file
    output_path = project_root / "results" / "position_counts.txt"
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Total players in database: {total_count}\n\n")
        f.write("Position counts:\n")
        f.write("-" * 30 + "\n")
        
        for position, count in position_counts.most_common():
            percentage = (count / total_count) * 100
            f.write(f"{position:<5}: {count:>5} ({percentage:.2f}%)\n")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
