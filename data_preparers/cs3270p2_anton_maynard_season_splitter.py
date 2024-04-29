#pylint: disable=C0301,W0718

"""
This script splits a CSV file containing data for multiple 
seasons into separate CSV files for each season.
The script reads each CSV file in the current directory, groups the 
data by season, and saves the grouped data into separate CSV files.

Functions:
- process_csv_file(csv_file): Processes a single CSV file by grouping 
the data by season and saving it into separate CSV files.
- main(): Entry point of the script. Reads all CSV files in the current 
directory and processes them concurrently using thread pool executor.

Usage:
1. Place the script in the same directory as the CSV files you want to split.
2. Run the script.

Note: The script requires the pandas library to be installed.
"""

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

time_start = time.time()

def process_csv_file(csv_file):
    """
    Processes a single CSV file by grouping the data by 
    season and saving it into separate CSV files.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        None
    """
    try:
        df = pd.read_csv(csv_file)
        if 'Season' not in df.columns:
            print(f"Warning: 'Season' column not found in {csv_file}. Skipping...")
            return

        for season, group in df.groupby('Season'):
            year = str(season)
            if not year.isdigit() or len(year) != 4:
                print(f"Warning: Invalid year '{year}' found in {csv_file}. Skipping groups with this year...")
                continue

            dir_name = f"./{year}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            new_filename = f"{dir_name}/{os.path.splitext(os.path.basename(csv_file))[0]}_{year}.csv"
            group.to_csv(new_filename, index=False)
        print(f"Processed {csv_file} into {new_filename}")
        print(f"Time taken: {time.time() - time_start} seconds")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    """
    Entry point of the script. Reads all CSV files in the current directory 
    and processes them concurrently using thread pool executor.

    Returns:
        None
    """
    csv_files = glob.glob('./*.csv')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_csv_file, csv_file) for csv_file in csv_files]
        for future in as_completed(futures):
            # Just to catch any potential exceptions and indicate completion
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
    print(f"Total time: {time.time() - time_start} seconds")

if __name__ == "__main__":
    main()
