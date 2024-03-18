import pandas as pd
import os
import glob
import shutil

def check_and_move_csv_files():
    # Ensure the 'parsed' directory exists
    parsed_dir = 'parsed'
    os.makedirs(parsed_dir, exist_ok=True)

    # Find all CSV files in the current directory
    csv_files = glob.glob('./*.csv')
    parsed_files = []

    for csv_file in csv_files:
        try:
            # Attempt to read the CSV file
            df = pd.read_csv(csv_file)
            # Check if the 'Season' column exists
            if 'Season' in df.columns:
                # Move file to 'parsed' directory
                shutil.move(csv_file, os.path.join(parsed_dir, os.path.basename(csv_file)))
                # Record the moved file's name
                parsed_files.append(os.path.basename(csv_file))
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Write the names of parsed files to "Parsed.txt"
    with open("Parsed.txt", "w") as file:
        for filename in parsed_files:
            file.write(f"{filename}\n")

if __name__ == "__main__":
    check_and_move_csv_files()
