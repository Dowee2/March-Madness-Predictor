#!/usr/bin/env python3

#pylint: disable=W0718,W0621,E0401,C0301,R0914,W0511

"""
This script predicts the winners of the tournament bracket in March Madness using pre-trained models. 
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd

import cs3270p1_the_overfitting_overlords_train_save_load_all_models as tsm

SEASON = 2024
def concat_stats_variants():
    """
    Returns the teams final stats calculation before the tournament for each season.

    Returns:
    - dict: A dictionary containing the different variants of the all team stats.
    """

    data_variants = {}
    data_variants['avg_10_df'] = pd.read_csv(f'data/Mens/Season/{SEASON}/MRegularSeasonDetailedResults_{SEASON}_avg_10_games.csv').groupby('TeamID').last().reset_index()
    data_variants['avg_rating_df'] = pd.read_csv(f'data/Mens/Season/{SEASON}/MRegularSeasonDetailedResults_{SEASON}_avg_w_rating.csv').groupby('TeamID').last().reset_index()
    data_variants['avg_df'] = pd.read_csv(f'data/Mens/Season/{SEASON}/MRegularSeasonDetailedResults_{SEASON}_avg.csv').groupby('TeamID').last().reset_index()
    data_variants['rating_rol10_df'] = pd.read_csv(f'data/Mens/Season/{SEASON}/MRegularSeasonDetailedResults_{SEASON}_avg_10_w_rating.csv').groupby('TeamID').last().reset_index()
    return data_variants

def build_game_matchups(bracket_team_matchups, team_stats):
    """
    Builds matchup data for the tournament bracket.

    Parameters:
    - bracket_team_matchups (DataFrame): The DataFrame containing the current matchups for a bracket.

    Returns:
    - DataFrame: A DataFrame where each row contains team_1's and team_2's stats for that specific matchup.
    """
    matchups = []
    for _, row in bracket_team_matchups.iterrows():
        team_1 = row['StrongSeed']
        team_2 = row['WeakSeed']
        team_1_data = team_stats[team_stats['TeamID'] == team_1]
        team_2_data = team_stats[team_stats['TeamID'] == team_2]
        team_1_data.columns = [f'team_1_{col}' for col in team_1_data.columns]
        team_2_data.columns = [f'team_2_{col}' for col in team_2_data.columns]
        team_1_data = team_1_data.reset_index(drop=True)
        team_2_data = team_2_data.reset_index(drop=True)
        matchup_data = pd.concat([team_1_data, team_2_data], axis=1)
        matchup_data['Slot'] = row['Slot']
        matchups.append(matchup_data)
    return pd.concat(matchups, axis=0)

def preprocess_playins(seeds_df):
    """
    Preprocess the play in teams by removing 'a' and 'b' designations and preparing strong and weak seeds.
    
    Parameters:
    - seeds_df (DataFrame): The DataFrame containing tournament seeds data, including 'Seed', 'TeamID' columns.
    
    Returns:
    - DataFrame: Processed DataFrame with 'StrongSeed' and 'WeakSeed' for play-in teams.
    """
    playin_teams = seeds_df[seeds_df['Seed'].str.contains('a') | seeds_df['Seed'].str.contains('b')].copy()
    playin_teams['Seed'] = playin_teams['Seed'].str.extract('([0-9A-Z]+)')
    playin_teams_match_df = playin_teams.groupby('Seed')['TeamID'].apply(list).reset_index()
    playin_teams_match_df['StrongSeed'] = playin_teams_match_df['TeamID'].apply(lambda x: x[0])
    playin_teams_match_df['WeakSeed'] = playin_teams_match_df['TeamID'].apply(lambda x: x[1])
    playin_teams_match_df.rename(columns={'Seed': 'Slot'}, inplace=True)
    return playin_teams_match_df.drop(columns='TeamID')

def predict_bracket_winners(bracket_matchups, model, scalar, drop_columns):
    """
    Predict the winners in the lower bracket using a pre-trained model and scaler.
    
    Parameters:
    - bracket_matchups (DataFrame): DataFrame of matchups in the lower bracket, excluding 'Seed' from scaling.
    - model (Model): Pre-trained prediction model.
    - scalar (Scaler): Pre-fitted scaler object for normalizing data.
    
    Returns:
    - DataFrame: Lower bracket DataFrame with an additional column 'team_1_won' indicating the predicted winner.
    """

    bracket_matchups = bracket_matchups.drop(columns=drop_columns)
    lower_bracket_scaled = scalar.fit_transform(bracket_matchups.drop(columns=['Slot']))
    bracket_matchups['team_1_won'] = model.predict(lower_bracket_scaled)
    return bracket_matchups

def update_seeds_with_winners(bracket, seeds_df):
    """
    Update the seeds DataFrame with the winners from the lower bracket predictions.
    
    Parameters:
    - bracket (DataFrame): The lower bracket DataFrame with predictions.
    - seeds_df (DataFrame): The original seeds DataFrame to be updated with current teams seedings.
    
    Returns:
    - DataFrame: Updated seeds DataFrame with winners.
    """
    bracket_winners = {}
    for _, row in bracket.iterrows():
        slot = row['Slot']
        if slot not in bracket_winners:
            bracket_winners[slot] = []
        bracket_winners[slot].append(row['team_1_TeamID'] if row['team_1_won'] == 1 else row['team_2_TeamID'])
    for curr_seed, team in bracket_winners.items():
        seeds_df.loc[len(seeds_df.index)] = [curr_seed, team[0]]
    seeds_df.sort_values(by='Seed', inplace=True)
    return seeds_df

def build_round_matchups(team_seeds, round_slots):
    """
    Build the Tourney round matchups based on seeds and updates team slots.
    
    Parameters:
    - team_seeds (DataFrame): DataFrame containing the seeds and corresponding team IDs.
    - round_slots (DataFrame): DataFrame containing the slots for the tournament matchups.
    
    Returns:
    - DataFrame: First-round matchups with updated team slots based on seeds.
    """

    for index, row in round_slots.iterrows():
        strong_team = team_seeds[(team_seeds['Seed'] == row['StrongSeed'])]['TeamID'].values[0]
        weak_team = team_seeds[(team_seeds['Seed'] == row['WeakSeed'])]['TeamID'].values[0]
        round_slots.at[index, 'StrongSeed'] = strong_team
        round_slots.at[index, 'WeakSeed'] = weak_team

    return round_slots

def predict_tourney_bracket(stats_columns, classifier):
    """
    Predict the winners of a tournament bracket in March Madness.

    Parameters:
    - stats_columns (dict): A dictionary containing the team stats columns.
    - classifier (Model): The pre-trained prediction model.

    Returns:
    - tuple: A tuple containing the matchups and the updated seeds DataFrame.
    """
    tourney_seeds_df = pd.read_csv(f'data/Mens/Season/{SEASON}/MNCAATourneySeeds_{SEASON}.csv')
    tourney_seeds_df.drop(columns=['Season'], inplace=True)
    tourney_slots_df = pd.read_csv(f'data/Mens/Season/{SEASON}/MNCAATourneySlots_{SEASON}.csv')

    team_stats = stats_columns['data'].dropna(axis=1, how='any')
    model = classifier
    drop_columns = stats_columns['drop_columns']
    scalar = StandardScaler()

    playin_teams_match_df = preprocess_playins(tourney_seeds_df)
    bracket_matchups = build_game_matchups(playin_teams_match_df, team_stats)
    bracket_matchups = predict_bracket_winners(bracket_matchups, model, scalar, drop_columns)
    tourney_seeds_df = update_seeds_with_winners(bracket_matchups, tourney_seeds_df)

    rounds = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

    tournament = []
    for current_round in rounds:
        curr_round_slots = tourney_slots_df[tourney_slots_df['Slot'].str.contains(current_round)]

        round_matchups = build_round_matchups(tourney_seeds_df,curr_round_slots)
        tournament.append(round_matchups)
        current_round_bracket = build_game_matchups(round_matchups, team_stats)
        current_round_bracket = predict_bracket_winners(current_round_bracket, model, scalar, drop_columns)
        tourney_seeds_df = update_seeds_with_winners(current_round_bracket, tourney_seeds_df)
    return tournament , tourney_seeds_df


def export_bracket_results(bracket_matchups, seeds):
    """
    Export the bracket results to a CSV file.
    
    Parameters:
    - bracket_matchups (dict): A dictionary containing the bracket matchups.
    - seeds (dict): A dictionary containing the seeds for the bracket.
    """
    predictions_df = pd.DataFrame()
    keys = bracket_matchups.keys()

    for variant in keys:
        length = len(bracket_matchups[variant])
        for i in range(length):
            classifier_name = list(bracket_matchups[variant][i].keys())[0]
            matchups = list(bracket_matchups[variant][i].values())[0]
            curr_tourney = pd.concat(matchups[0:])
            curr_seed = list(seeds[variant][i].values())[0]

            merged_df = match_brackets_seeds(variant, classifier_name, curr_tourney, curr_seed)
            predictions_df = pd.concat([predictions_df, merged_df], axis=1)

    predictions_df = predictions_df.transpose()
    convert_to_team_names = True
    if convert_to_team_names:
        convert_teamid_to_names(predictions_df)
    else:
        predictions_df.to_csv(f'MNCAATourneyPredictionsTest_{SEASON}.csv', index=False)

def convert_teamid_to_names(predictions_df):
    """
    Converts all TeamID to TeamName in the predictions DataFrame.
    """

    teams_names_df = pd.read_csv('data/Mens/MTeams.csv')
    team_names_map = teams_names_df.set_index('TeamID')['TeamName'].to_dict()
    for col in predictions_df.columns:
        predictions_df[col] = predictions_df[col].map(team_names_map)

    predictions_df.to_csv(f'MNCAATourneyPredictionsTest_{SEASON}.csv', index=True)


def match_brackets_seeds(variant, classifier_name, tourney, seed):
    """
    Match the bracket matchups with the corresponding seeds.
    """
    identifier = classifier_name + '_' + variant
    tourney.rename(columns = {'StrongSeed' : 'StrongSeed_'+ identifier, 'WeakSeed' : 'WeakSeed_'+ identifier, 'Slot': 'Seed'}, inplace=True)
    tourney.drop(columns ='Season', inplace=True)
    tourney.set_index('Seed', inplace=True)

    identifier = classifier_name + '_' + variant
    seed.rename(columns = {'TeamID' : 'Winner_'+ identifier}, inplace=True)
    seed.set_index('Seed', inplace=True)

    merged_df = pd.merge(tourney, seed, left_on='Seed', right_on='Seed', how='inner')
    return merged_df


if __name__ == '__main__':
    tournament_preds = {}
    tourney_seeds = {}
    data = concat_stats_variants()

    trained_models = tsm.load_models()

    avg_10_models = trained_models['avg_10']
    averaged_10_data = {'data': data['avg_10_df'],'drop_columns': ['team_1_DayNum', 'team_1_Week','team_2_DayNum', 'team_2_Week']}
    tournament_preds['avg_10'] = []
    tourney_seeds['avg_10'] = []
    for model in avg_10_models.values():
        tournament, seeds = predict_tourney_bracket(averaged_10_data, model)
        tournament_preds['avg_10'].append({type(model).__name__ : tournament})
        tourney_seeds['avg_10'].append({type(model).__name__ : seeds})

    avg_models = trained_models['avg']
    averaged_season_data = {'data': data['avg_df'],'drop_columns': []}
    tourney_seeds['avg'] = []
    tournament_preds['avg'] = []
    for model in avg_models.values():
        tournament, seeds = predict_tourney_bracket(averaged_season_data, model)
        tournament_preds['avg'].append({type(model).__name__ : tournament})
        tourney_seeds['avg'].append({type(model).__name__ : seeds})

    rating_rol10_models = trained_models['rating_rol10']
    rating_roll10_data = {'data': data['rating_rol10_df'],'drop_columns': [ 'team_1_DayNum',
             'team_1_Week_x', 'team_1_Week_y', 'team_2_DayNum',
             'team_2_Week_x','team_2_Week_y',]}
    tournament_preds['rating_rol10'] = []
    tourney_seeds['rating_rol10'] = []
    for model in rating_rol10_models.values():
        tournament, seeds = predict_tourney_bracket(rating_roll10_data, model)
        tournament_preds['rating_rol10'].append({type(model).__name__ : tournament})
        tourney_seeds['rating_rol10'].append({type(model).__name__ : seeds})

    avg_rating_models = trained_models['avg_rating']
    avg_rating_data = {'data': data['avg_rating_df'],'drop_columns': []}
    tourney_seeds['avg_rating'] = []
    tournament_preds['avg_rating'] = []
    for model in avg_rating_models.values():
        tournament, seeds = predict_tourney_bracket(avg_rating_data, model)
        tournament_preds['avg_rating'].append({type(model).__name__ : tournament})
        tourney_seeds['avg_rating'].append({type(model).__name__ : seeds})

    export_bracket_results(tournament_preds, tourney_seeds)
