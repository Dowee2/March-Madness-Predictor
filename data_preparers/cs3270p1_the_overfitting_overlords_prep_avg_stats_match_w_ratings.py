
#pylint: disable=C0114,C0301,W0718,C0103

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
    df = calculate_additional_stats(df)
   # Stats when the team wins
    win_stats = df[['Season','WTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM',
                    'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    win_stats.columns = ['Season','TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM',
                         'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it wins (opponents' performance)
    win_against_stats = df[['Season','WTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM',
                            'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    win_against_stats.columns = ['Season','TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3',
                                  'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 
                                  'Blk', 'PF']

    # Stats when the team loses
    lose_stats = df[['Season','LTeamID', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                     'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    lose_stats.columns = ['Season','TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA',
                          'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

    # Stats against the team when it loses (opponents' performance)
    lose_against_stats = df[['Season','LTeamID', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM',
                             'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    lose_against_stats.columns = ['Season','TeamID', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3',
                                  'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 
                                  'Blk', 'PF']    

    # Combine winning and losing stats
    all_stats = pd.concat([win_stats, lose_stats])
    all_against_stats = pd.concat([win_against_stats, lose_against_stats])

    # Calculate the mean for stats and stats against separately
    avg_stats = all_stats.groupby(['Season','TeamID']).mean().reset_index()
    avg_against_stats = all_against_stats.groupby(['Season','TeamID']).mean().reset_index()

    # Merge the average stats with average stats against
    avg_merged_stats = pd.merge(avg_stats, avg_against_stats, on=['Season','TeamID'], suffixes=('', '_A'))
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

    ordinal_df = ordinal_df.sort_values(by=['Season','TeamID']).reset_index(drop=True)
    ordinal_df = ordinal_df.drop(columns=['RankingDayNum'])

    system_names = ordinal_df['SystemName'].unique()
    teams_names = ordinal_df['TeamID'].unique()
    system_no_rank_all_teams = []

    for system in system_names:
        teams_in_system = ordinal_df[ordinal_df['SystemName'] == system]['TeamID'].unique()
        if len(teams_in_system) != len(teams_names):
            system_no_rank_all_teams.append(system)

    ordinal_df = ordinal_df[~ordinal_df['SystemName'].isin(system_no_rank_all_teams)]
    ordinal_pivot = ordinal_df.pivot_table(index=['Season','TeamID'], columns='SystemName',
                                           values='OrdinalRank').reset_index()
    ordinal_pivot.sort_values(by=['Season','TeamID'])
    ordinal_pivot = ordinal_pivot.ffill()
    ordinal_pivot = ordinal_pivot.groupby(['Season','TeamID']).apply(lambda x: x.interpolate(method='linear', limit_direction='both')).reset_index(drop=True)
    ordinal_pivot = ordinal_pivot.groupby(['Season','TeamID']).mean().reset_index()
    
    return ordinal_pivot

def merge_ratings_stats(ordinal_df, teams_stats_avg_df):
    """
    Merge the ordinal dataframe and the teams' weekly stats 
    dataframe based on the 'TeamID' and 'DayNum' columns.
    
    Args:
        ordinal_df (pandas.DataFrame): The ordinal dataframe containing the team ratings.
        teams_stats_weekly_df (pandas.DataFrame): The teams' weekly stats dataframe.
        
    Returns:
        pandas.DataFrame: The merged dataframe with the team ratings and weekly stats.
    """
    ordinal_df = ordinal_df.sort_values(by=['TeamID'])

    teams_stats_avg_df = teams_stats_avg_df.sort_values(by=['Season','TeamID'])
    avg_stats_w_rating = pd.merge(teams_stats_avg_df, ordinal_df, on=['Season','TeamID'], suffixes=('', '_A'))
    avg_stats_w_rating = avg_stats_w_rating.sort_values(by=['Season','TeamID']).reset_index(drop=True)
    rank_columns = avg_stats_w_rating.columns[34:]

    # Fill the NaN values in the rank columns with the previous values for each team. If still NaN, then drop the column.
    avg_stats_w_rating[rank_columns] = avg_stats_w_rating.groupby(['Season','TeamID'])[rank_columns].bfill()
    avg_stats_w_rating = avg_stats_w_rating.dropna(axis = 1, how= 'any')
    return avg_stats_w_rating


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
        team_1_stats = stats.loc[(stats['TeamID'] == team_1)].add_prefix('team_1_').iloc[-1]
        team_2_stats = stats.loc[(stats['TeamID'] == team_2)].add_prefix('team_2_').iloc[-1]

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

    games_df = pd.DataFrame()
    ordinal_df = pd.DataFrame()

    for season in range(2003 , 2025):
        season_games = pd.read_csv(f'{data_location}/{season}/MRegularSeasonDetailedResults_{season}.csv')
        games_df = pd.concat([games_df, season_games])
        ordinal_games = pd.read_csv(f'{data_location}/{season}/MMasseyOrdinals_{season}.csv')
        ordinal_df = pd.concat([ordinal_df, ordinal_games])
        
    avg_stats = prepare_team_stats(games_df)
    ordinal_df = prep_ordinal_ratings_for_merge(ordinal_df)
    avg_stats_w_rating = merge_ratings_stats(ordinal_df, avg_stats)

    for season in avg_stats_w_rating['Season'].unique():
        season_stats = avg_stats_w_rating[avg_stats_w_rating['Season'] == season]
        season_games = games_df[games_df['Season'] == season]
        season_stats.to_csv(f'{data_location}/{season}/MRegularSeasonDetailedResults_{season}_avg_w_rating.csv', index=False)
        prepared_matches = prepare_matchup_data(season_games, season_stats)
        prepared_matches.to_csv(f'{data_location}/{season}/MRegularSeasonDetailedResults_{season}_matchups_avg_w_rating.csv', index=False)


    # for season in seasons:
    #     currdir = os.path.join(data_location, season)
    #     try:
    #         df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}.csv')

    #         ordinal_df = pd.read_csv(f'{currdir}/MMasseyOrdinals_{season}.csv')
    #         ordinal_df = prep_ordinal_ratings_for_merge(ordinal_df)
    #         avg_stats_w_rating = merge_ratings_stats(ordinal_df, avg_stats)

    #         stats_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_avg_w_rating.csv'
    #         if os.path.exists(stats_path):
    #             os.remove(stats_path)
    #         avg_stats_w_rating.to_csv(stats_path, index=False)

    #         prepared_matches = prepare_matchup_data(df, avg_stats_w_rating)
    #         prepared_path = f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg_w_rating.csv'
    #         if os.path.exists(prepared_path):
    #             os.remove(prepared_path)
    #         prepared_matches.to_csv(prepared_path, index=False)
    #     except Exception as e:
    #         print(f"Error processing {season}: {e}")

if __name__ == "__main__":
    main()