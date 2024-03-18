import pandas as pd
import os

dir = 'data/Mens/Season/'

#Get the list of all Directories in the Season Directory
seasons = os.listdir(dir)
for season in seasons:
    currdir = os.path.join(dir, season)
    try:
        df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}.csv')

        # Calculate additional stats
        df['WFGM2'] = df['WFGM'] - df['WFGM3']
        df['WFGA2'] = df['WFGA'] - df['WFGA3']
        df['LFGM2'] = df['LFGM'] - df['LFGM3']
        df['LFGA2'] = df['LFGA'] - df['LFGA3']

        # Create a DataFrame for winning team stats and rename columns to remove 'W' and 'L' prefixes
        win_stats = df[['WTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
        win_stats.columns = ['TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

        # Create a DataFrame for losing team stats and rename columns to include 'N' prefix for negatives
        lose_stats = df[['LTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
        lose_stats.columns = ['TeamID', 'NFGM', 'NFGA', 'NFGM2', 'NFGA2', 'NFGM3', 'NFGA3', 'NFTM', 'NFTA', 'NOR', 'NDR', 'NAst', 'NTO', 'NStl', 'NBlk', 'NPF']

        # Concatenate win and lose stats into one DataFrame
        all_stats = pd.concat([win_stats, lose_stats])

        # Group by TeamID and calculate mean for each stat
        avg_stats = all_stats.groupby('TeamID').mean().reset_index()
        
        #limit the decimal places to 2
        avg_stats = avg_stats.round(2)

        # Display the first few rows to check
        if os.path.exists(f'{currdir}/MRegularSeasonDetailedResults_{season}_avg.csv'):
            os.remove(f'{currdir}/MRegularSeasonDetailedResults_{season}_avg.csv')
            
        avg_stats.to_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_avg.csv', index=False)
    except Exception as e:
        print(f"Error processing {season}: {e}")
