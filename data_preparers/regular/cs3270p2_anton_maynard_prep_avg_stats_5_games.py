
#pylint: disable=C0301,C0114,W0718,C0103

import os
import pandas as pd
import numpy as np


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
    df['Week'] = (df['DayNum']-1)/7 +1
    df['Week'] = df['Week'].apply(np.floor)
    return df

def prepare_team_stats(df):
    """
    Prepares and aggregates team statistics and statistics against from game results.

    Parameters:
    - df (DataFrame): The game results DataFrame with additional stats.

    Returns:
    - DataFrame: A DataFrame with average stats per team and stats against.
    """
    df = calculate_additional_stats(df)
    # Stats when the team wins
    win_stats = df[['WTeamID','DayNum','Week', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3',
                    'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    win_stats.columns = ['TeamID','DayNum','Week', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3',
                         'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it wins (opponents' performance)
    win_against_stats = df[['WTeamID','DayNum','Week', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3',
                            'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk',
                            'LPF']].copy()
    win_against_stats.columns = ['TeamID','DayNum','Week', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A',
                                 'FGM3A', 'FGA3A', 'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA',
                                 'TOA', 'StlA', 'BlkA', 'PFA']

    # Stats when the team loses
    lose_stats = df[['LTeamID','DayNum','Week', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3',
                     'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk',
                     'LPF']].copy()
    lose_stats.columns = ['TeamID', 'DayNum','Week','FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3',
                          'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it loses (opponents' performance)
    lose_against_stats = df[['LTeamID','DayNum','Week','WFGM', 'WFGA', 'WFGM2', 'WFGA2',
                             'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
                             'WStl', 'WBlk', 'WPF']].copy()
    lose_against_stats.columns = ['TeamID','DayNum','Week', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A',
                                  'FGM3A', 'FGA3A', 'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA',
                                  'TOA', 'StlA', 'BlkA', 'PFA']

    # Combine winning and losing stats
    all_stats = pd.concat([win_stats, lose_stats]).sort_values(by=['TeamID', 'DayNum'])
    all_against_stats = pd.concat([win_against_stats,
                                   lose_against_stats]).sort_values(by=['TeamID', 'DayNum'])

    team_daynum_week_for =  all_stats[['TeamID', 'DayNum', 'Week']].reset_index(drop=True)
    team_daynum_week_against = all_against_stats[['TeamID',
                                                  'DayNum', 'Week']].reset_index(drop=True)

    all_stats.drop(columns=['Week', 'DayNum'], inplace=True)
    all_against_stats.drop(columns=['Week', 'DayNum'], inplace=True)

    stats_rolling = all_stats.groupby('TeamID').rolling(window=5,
                                                        min_periods=1).mean().reset_index(drop=True)
    against_rolling = all_against_stats.groupby('TeamID').rolling(window=5, min_periods=1).mean().reset_index(drop=True)

    stats_rolling = pd.concat([team_daynum_week_for, stats_rolling], axis=1)
    against_rolling = pd.concat([team_daynum_week_against, against_rolling], axis=1)

    merged_stats = pd.merge(stats_rolling, against_rolling, on=['TeamID','DayNum','Week'], suffixes=('', '_A'))
    return merged_stats


def prepare_matchup_data(games_df, stats):
    """
    Merges game data with team stats to prepare matchup data.

    Parameters:
    - games_df (DataFrame): The DataFrame containing game results.
    - avg_stats (DataFrame): The DataFrame containing average stats per team.

    Returns:
    - DataFrame: Matchup data with team stats and game outcome.
    """
    processed_data = []

    for _, row in games_df.iterrows():
        team_1, team_2 = sorted((row['WTeamID'], row['LTeamID']))
        team_1_won = 1 if team_1 == row['WTeamID'] else 0
        day = row['DayNum']
        team_1_stats = stats.loc[(stats['TeamID'] == team_1) & (stats['DayNum'] == day)].add_prefix('team_1_').iloc[-1]
        team_2_stats = stats.loc[(stats['TeamID'] == team_2) & (stats['DayNum'] == day)].add_prefix('team_2_').iloc[-1]


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
            teams_stats_weekly_df = prepare_team_stats(df)

            stats_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_avg_5_games.csv'
            if os.path.exists(stats_path):
                os.remove(stats_path)
            teams_stats_weekly_df.to_csv(stats_path, index=False)

            prepared_matches = prepare_matchup_data(df, teams_stats_weekly_df)
            prepared_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg_5.csv'
            if os.path.exists(prepared_path):
                os.remove(prepared_path)
            prepared_matches.to_csv(prepared_path, index=False)
        except Exception as e:
            print(f"Error processing {season}: {e}")

if __name__ == "__main__":
    main()
