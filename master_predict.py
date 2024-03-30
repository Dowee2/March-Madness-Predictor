import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import predict_game_winner_by_avg as avg
import predict_game_winner_by_avg_10 as avg_10
import predict_game_winner_by_avg_w_ratings as avg_rating
import predict_game_winner_10_games_w_rating as rating

season = 2023

def concat_stats_variants():
    """
    Returns the teams final stats calculation before the tournament for each season.
    """

    data_variants = {}
    data_variants['avg_10_df'] = [pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_10_games.csv').groupby('TeamID').last().reset_index()]
    data_variants['avg_rating_df'] = [pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_w_rating.csv').groupby('TeamID').last().reset_index()]
    data_variants['avg_df'] = [pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg.csv').groupby('TeamID').last().reset_index()]  
    data_variants['rating_rol10_df'] = [pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_10_w_rating.csv').groupby('TeamID').last().reset_index()]
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
    ####TODO: pass in the columns that are supposed to be dropped
    bracket_matchups = bracket_matchups.drop(columns=drop_columns)
    # bracket_matchups = bracket_matchups.drop(columns= ['team_1_DayNum', 'team_1_Week','team_2_DayNum', 'team_2_Week'])
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

def predict_tourney_bracket(stats_and_models):
    tourney_seeds_df = pd.read_csv(f'{season}/MNCAATourneySeeds_{season}.csv')
    tourney_seeds_df.drop(columns=['Season'], inplace=True)
    tourney_slots_df = pd.read_csv(f'{season}/MNCAATourneySlots_{season}.csv')

    team_stats = stats_and_models['data']
    model = stats_and_models['model']
    drop_columns = stats_and_models['drop_columns']
    scalar = StandardScaler()

    playin_teams_match_df = preprocess_playins(tourney_seeds_df)
    bracket_matchups = build_game_matchups(playin_teams_match_df, team_stats[season])
    bracket_matchups = predict_bracket_winners(bracket_matchups, model, scalar, drop_columns)
    tourney_seeds_df = update_seeds_with_winners(bracket_matchups, tourney_seeds_df)

    rounds = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

    matchups = []
    for current_round in rounds:
        curr_round_slots = tourney_slots_df[tourney_slots_df['Slot'].str.contains(current_round)]

        round_matchups = build_round_matchups(tourney_seeds_df,curr_round_slots)
        matchups.append(round_matchups)
        current_round_bracket = build_game_matchups(round_matchups, team_stats[season])
        current_round_bracket = predict_bracket_winners(current_round_bracket, model, scalar, drop_columns)
        tourney_seeds_df = update_seeds_with_winners(current_round_bracket, tourney_seeds_df)
    return matchups , tourney_seeds_df

def export_bracket_results(bracket_matchups, seeds):
    """
    Export the bracket results to a CSV file.
    
    Parameters:
    - matchups (DataFrame): DataFrame containing the bracket matchups.
    - seeds (DataFrame): DataFrame containing the seeds for the bracket.
    """
    for variant in bracket_matchups.keys():
        for classifier in bracket_matchups[variant].keys():
            matchups_df = pd.concat(bracket_matchups[variant][classifier], axis=0)
            matchups_df.to_csv(f'MNCAATourneyPredictions_matchups_{season}_{variant}_{classifier}.csv', index=False)
            seeds[variant][classifier].to_csv(f'MNCAATourneyPredictions__seeds_{season}_{variant}_{classifier}.csv', index=False)

if __name__ == '__main__':
    # df = pd.read_csv('data/Mens/Season/2022/MRegularSeasonDetailedResults_2022_matchups_avg_10.csv')
    # df = concat_seasons()

    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, activation = 'logistic', learning_rate = 'invscaling'),
        'Naive Bayes': GaussianNB()
    }
    matchups = {}
    tourney_seeds = {}
    data = concat_stats_variants()

    avg_10_models = avg_10.train_models(classifiers)
    averaged_10_data = {'data': data['avg_10_df'], 'model': avg_10_models, 'drop_columns': ['team_1_DayNum', 'team_1_Week','team_2_DayNum', 'team_2_Week']}
    for model in avg_10_models:
        matchup, seeds = predict_tourney_bracket(averaged_10_data)
        matchups['avg_10'][type.__name__] = [matchup]
        tourney_seeds['avg_10'][type.__name__] = [seeds]
    
    avg_models = avg.train_models(classifiers)
    averaged_season_data = {'data': data['avg_df'], 'model': avg_models, 'drop_columns': []}
    for model in avg_models:
        matchup, seeds = predict_tourney_bracket(averaged_season_data)
        matchups['avg'][type.__name__] = [matchup]
        tourney_seeds['avg'][type.__name__] = [seeds]

    
    rating_rol10_models = rating.train_models(classifiers)
    rating_roll10_data = {'data': data['rating_rol10_df'], 'model': rating_rol10_models, 'drop_columns': ['team_1_DayNum', 'team_1_Week','team_2_DayNum', 'team_2_Week']}
    for model in rating_rol10_models:
        matchup, seeds = predict_tourney_bracket(rating_roll10_data)
        matchups['rating_roll10'][type.__name__] = [matchup]
        tourney_seeds['rating_roll10'][type.__name__] = [seeds]


    # avg_rating_models = avg_rating.train_models(classifiers)
    # avg_rating_data = {'data': data['avg_rating_df'], 'model': avg_rating_models, 'drop_columns': []}
    # for model in avg_rating_models:
    #     matchup, seeds = predict_tourney_bracket(avg_rating_data)
    #     matchups['avg_rating'][type.__name__] = [matchup]
    #     tourney_seeds['avg_rating'][type.__name__] = [seeds]
    
    export_bracket_results(matchups, tourney_seeds)
    