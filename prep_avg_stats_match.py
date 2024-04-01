
#pylint: disable=W0718,W0621,E0401,C0301,R0914,W0718,C0114

import os
import pandas as pd


def calculate_additional_stats(df):
    """
    Adds calculated statistics for two-point field goals to the DataFrame.

    Parameters:
    - df (DataFrame): The original game results DataFrame.

    Returns:
    - DataFrame: The modified DataFrame with additional stats.
    """
    df['WFGM2'] = df['WFGM'] - df['WFGM3']
    df['WFGA2'] = df['WFGA'] - df['WFGA3']
    df['LFGM2'] = df['LFGM'] - df['LFGM3']
    df['LFGA2'] = df['LFGA'] - df['LFGA3']
    return df

def prepare_team_stats(df):
    """
    Prepares and aggregates team statistics and statistics against from game results.

    Parameters:
    - df (DataFrame): The game results DataFrame with additional stats.

    Returns:
    - DataFrame: A DataFrame with average stats per team and stats against.
    """
    # Stats when the team wins
    win_stats = df[['WTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM',
                    'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    win_stats.columns = ['TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM',
                         'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it wins (opponents' performance)
    win_against_stats = df[['WTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM',
                            'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    win_against_stats.columns = ['TeamID', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A', 'FGM3A', 'FGA3A',
                                 'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA', 'TOA', 'StlA', 'BlkA', 'PFA']

    # Stats when the team loses
    lose_stats = df[['LTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                     'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    lose_stats.columns = ['TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM',
                          'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it loses (opponents' performance)
    lose_against_stats = df[['LTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM',
                             'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    lose_against_stats.columns = ['TeamID', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A', 'FGM3A', 'FGA3A',
                                  'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA', 'TOA', 'StlA',
                                  'BlkA', 'PFA']

    # Combine winning and losing stats
    all_stats = pd.concat([win_stats, lose_stats])
    all_against_stats = pd.concat([win_against_stats, lose_against_stats])

    # Calculate the mean for stats and stats against separately
    avg_stats = all_stats.groupby('TeamID').mean().reset_index()
    avg_against_stats = all_against_stats.groupby('TeamID').mean().reset_index()

    # Merge the average stats with average stats against
    avg_merged_stats = pd.merge(avg_stats, avg_against_stats, on='TeamID', suffixes=('', '_A'))
    avg_merged_stats = avg_merged_stats.round(2)
    return avg_merged_stats

def prepare_matchup_data(games_df, avg_stats):
    """
    Merges game data with team stats to prepare matchup data.

    Parameters:
    - games_df (DataFrame): The DataFrame containing game results.
    - avg_stats (DataFrame): The DataFrame containing average stats per team.

    Returns:
    - DataFrame: Matchup data with team stats and game outcome.
    """
    avg_stats.set_index('TeamID', inplace=True)
    processed_data = []

    for _, row in games_df.iterrows():
        team_1, team_2 = sorted((row['WTeamID'], row['LTeamID']))
        team_1_won = 1 if team_1 == row['WTeamID'] else 0
        team_1_stats = avg_stats.loc[team_1].add_prefix('team_1_').to_dict()
        team_2_stats = avg_stats.loc[team_2].add_prefix('team_2_').to_dict()

        matchup_data = {
            'Season': row['Season'],
            'DayNum': row['DayNum'],
            'team_1': team_1,
            'team_2': team_2,
            'team_1_won': team_1_won
        }
        matchup_data.update(team_1_stats)
        matchup_data.update(team_2_stats)

        processed_data.append(matchup_data)

    return pd.DataFrame(processed_data)


def main():
    """
    The main method that starts the program.
    """
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)

    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}.csv')
            df = calculate_additional_stats(df)
            avg_stats = prepare_team_stats(df)

            stats_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_avg.csv'
            if os.path.exists(stats_path):
                os.remove(stats_path)

            avg_stats.to_csv(stats_path, index=False)

            prepared_matches = prepare_matchup_data(df, avg_stats)
            prepared_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg.csv'
            if os.path.exists(prepared_path):
                os.remove(prepared_path)
            prepared_matches.to_csv(prepared_path, index=False)
        except Exception as e:
            print(f"Error processing {season}: {e}")

if __name__ == "__main__":
    main()
