import pandas as pd
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

time_start = time.time()

def process_csv_file(csv_file):
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
