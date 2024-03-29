import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import predict_game_winner_by_avg as avg
import predict_game_winner_by_avg_10 as avg_10
import predict_game_winner_by_avg_w_ratings as avg_rating
import predict_game_winner_10_games_w_rating as rating


def concat_stats_variants():
    data_variants = {}
    for season in range(2003 , 2025):
        data_variants['avg_10_df'].append(pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_10_games.csv'))
        data_variants['avg_rating_df'].append(pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_w_rating.csv'))
        data_variants['avg_df'].append(pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_10.csv'))       
        data_variants['rating_rol10_df'].append(pd.read_csv(f'data/Mens/Season/{season}/MRegularSeasonDetailedResults_{season}_avg_10_games.csv'))
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

        all_team_1_data = team_stats[team_stats['TeamID'] == team_1]
        all_team_2_data = team_stats[team_stats['TeamID'] == team_2]
        team_1_data = all_team_1_data[all_team_1_data['DayNum'] == all_team_1_data['DayNum'].max()]
        team_2_data = all_team_2_data[all_team_2_data['DayNum'] == all_team_2_data['DayNum'].max()]
        
        
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

def predict_bracket_winners(bracket_matchups, model, scalar):
    """
    Predict the winners in the lower bracket using a pre-trained model and scaler.
    
    Parameters:
    - bracket_matchups (DataFrame): DataFrame of matchups in the lower bracket, excluding 'Seed' from scaling.
    - model (Model): Pre-trained prediction model.
    - scalar (Scaler): Pre-fitted scaler object for normalizing data.
    
    Returns:
    - DataFrame: Lower bracket DataFrame with an additional column 'team_1_won' indicating the predicted winner.
    """
    bracket_matchups = bracket_matchups.drop(columns= ['team_1_DayNum', 'team_1_Week','team_2_DayNum', 'team_2_Week'])
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

def predict_tourney_bracket(team_stats, model):
    tourney_seeds_df = pd.read_csv('2023/MNCAATourneySeeds_2023.csv')
    tourney_seeds_df.drop(columns=['Season'], inplace=True)
    tourney_slots_df = pd.read_csv('2023/MNCAATourneySlots_2023.csv')

    scalar = StandardScaler()

    playin_teams_match_df = preprocess_playins(tourney_seeds_df)
    bracket_matchups = build_game_matchups(playin_teams_match_df, team_stats[2023])
    bracket_matchups = predict_bracket_winners(bracket_matchups, model, scalar)
    tourney_seeds_df = update_seeds_with_winners(bracket_matchups, tourney_seeds_df)

    rounds = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

    matchups = []
    for current_round in rounds:
        curr_round_slots = tourney_slots_df[tourney_slots_df['Slot'].str.contains(current_round)]

        round_matchups = build_round_matchups(tourney_seeds_df,curr_round_slots)
        matchups.append(round_matchups)
        current_round_bracket = build_game_matchups(round_matchups, team_stats[2023])
        current_round_bracket = predict_bracket_winners(current_round_bracket, model, scalar)
        tourney_seeds_df = update_seeds_with_winners(current_round_bracket, tourney_seeds_df)


if __name__ == '__main__':
    # df = pd.read_csv('data/Mens/Season/2022/MRegularSeasonDetailedResults_2022_matchups_avg_10.csv')
    # df = concat_seasons()

    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, activation = 'logistic', learning_rate = 'invscaling')
    }

    avg_10_models = avg_10.train_models(classifiers)
    avg_models = avg.train_models(classifiers)
    avg_rating_models = avg_rating.train_models(classifiers)
    rating_models = rating.train_models(classifiers)

    data = concat_stats_variants()
    avg_10_data = data['avg_10_df']
    avg_data = data['avg_df']
    avg_rating_data = data['avg_rating_df']
    rating_data = data['rating_rol10_df']

    for model in avg_10_models:
        predict_tourney_bracket(avg_10_data, avg_10_models[model])
    
    for model in avg_models:
        predict_tourney_bracket(avg_data, avg_models[model])
    
    for model in avg_rating_models:
        predict_tourney_bracket(avg_rating_data, avg_rating_models[model])
    
    for model in rating_models:
        predict_tourney_bracket(rating_data, rating_models[model])
    
