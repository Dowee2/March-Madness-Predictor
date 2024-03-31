#!/usr/bin/env python3

#pylint: disable=W0718,W0621,E0401,C0301,R0914

import pandas as pd
import os

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
    win_stats = df[['WTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    win_stats.columns = ['TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    
    # Stats against the team when it wins (opponents' performance)
    win_against_stats = df[['WTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    win_against_stats.columns = ['TeamID', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A', 'FGM3A', 'FGA3A', 'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA', 'TOA', 'StlA', 'BlkA', 'PFA']

    # Stats when the team loses
    lose_stats = df[['LTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    lose_stats.columns = ['TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    
    # Stats against the team when it loses (opponents' performance)
    lose_against_stats = df[['LTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    lose_against_stats.columns = ['TeamID', 'FGMA', 'FGAA', 'FGM2A', 'FGA2A', 'FGM3A', 'FGA3A', 'FTMA', 'FTAA', 'ORA', 'DRA', 'AstA', 'TOA', 'StlA', 'BlkA', 'PFA']

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


def prep_ordinal_ratings_for_merge(ordinal_df):
    """
    Preprocesses the ordinal ratings dataframe for merging with other dataframes.
    
    Args:
        ordinal_df (pandas.DataFrame): The ordinal ratings dataframe.
        
    Returns:
        pandas.DataFrame: The preprocessed ordinal ratings dataframe.
    """
    
    ordinal_df = ordinal_df.sort_values(by=['TeamID']).reset_index(drop=True)
    ordinal_df = ordinal_df.drop(columns=['RankingDayNum'])

    system_names = ordinal_df['SystemName'].unique()
    teams_names = ordinal_df['TeamID'].unique()
    system_no_rank_all_teams = []

    for system in system_names:
        teams_in_system = ordinal_df[ordinal_df['SystemName'] == system]['TeamID'].unique()
        if len(teams_in_system) != len(teams_names):
            system_no_rank_all_teams.append(system)

    ordinal_df = ordinal_df[~ordinal_df['SystemName'].isin(system_no_rank_all_teams)]
    ordinal_pivot = ordinal_df.pivot_table(index=['TeamID'], columns='SystemName', values='OrdinalRank').reset_index()
    ordinal_pivot.sort_values(by=['TeamID'])
    ordinal_pivot = ordinal_pivot.groupby('TeamID').mean().reset_index()

    return ordinal_pivot


def prepare_matchup_data(games_df, stats):
    """
    Merges game data with team stats to prepare matchup data.

    Parameters:
    - games_df (DataFrame): The DataFrame containing game results.
    - avg_stats (DataFrame): The DataFrame containing average stats per team.

    Returns:
    - DataFrame: Matchup data with team stats and game outcome.
    """
    stats.set_index('TeamID', inplace=True)
    processed_data = []

    for _, row in games_df.iterrows():
        team_1, team_2 = sorted((row['WTeamID'], row['LTeamID']))
        team_1_won = 1 if team_1 == row['WTeamID'] else 0
        team_1_stats = stats.loc[team_1].add_prefix('team_1_').iloc[-1]
        team_2_stats = stats.loc[team_2].add_prefix('team_2_').iloc[-1]
        
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

def merge_ratings_stats(ordinal_df, teams_stats_avg_df):
    """
    Merge the ordinal dataframe and the teams' weekly stats dataframe based on the 'TeamID' and 'DayNum' columns.
    
    Args:
        ordinal_df (pandas.DataFrame): The ordinal dataframe containing the team ratings.
        teams_stats_weekly_df (pandas.DataFrame): The teams' weekly stats dataframe.
        
    Returns:
        pandas.DataFrame: The merged dataframe with the team ratings and weekly stats.
    """
    ordinal_df = ordinal_df.sort_values(by=['TeamID'])

    teams_stats_avg_df = teams_stats_avg_df.sort_values(by=['TeamID'])
    avg_stats_w_rating = pd.merge(teams_stats_avg_df, ordinal_df, on='TeamID', suffixes=('', '_A'))
    avg_stats_w_rating = avg_stats_w_rating.sort_values(by=['TeamID']).reset_index(drop=True)
    rank_columns = avg_stats_w_rating.columns[34:]

    avg_stats_w_rating[rank_columns] = avg_stats_w_rating.groupby('TeamID')[rank_columns].bfill()
    avg_stats_w_rating = avg_stats_w_rating.dropna(axis = 1, how= 'any')
    return avg_stats_w_rating


def main():
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)

    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}.csv')
            df = calculate_additional_stats(df)
            avg_stats = prepare_team_stats(df)
            
            stats_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_avg_w_rating.csv'
            if os.path.exists(stats_path):
                os.remove(stats_path)
            avg_stats.to_csv(stats_path, index=False)

            ordinal_df = pd.read_csv(f'{currdir}/MMasseyOrdinals_{season}.csv')
            ordinal_df = prep_ordinal_ratings_for_merge(ordinal_df)
            avg_stats_w_rating = merge_ratings_stats(ordinal_df, avg_stats)

            prepared_matches = prepare_matchup_data(df, avg_stats_w_rating)
            prepared_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg_w_rating.csv'
            if os.path.exists(prepared_path):
                os.remove(prepared_path)
            prepared_matches.to_csv(prepared_path, index=False)
        except Exception as e:
            print(f"Error processing {season}: {e}")

if __name__ == "__main__":
    main()