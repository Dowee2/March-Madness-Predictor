#!/usr/bin/env python3
"""Module for training and evaluating different classifiers on basketball game data."""

#pylint: disable=W0718,W0621,E0401,C0301,R0914

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10),
    'Random Forest': RandomForestClassifier(
        random_state=3270, n_estimators=200, max_depth=10,
        min_samples_split=10, n_jobs=-1),
    'Logistic Regression': LogisticRegression(
        random_state=3270, max_iter=1000, penalty=None, solver='lbfgs'),
    'XGBoost': XGBClassifier(
        random_state=3270, n_estimators=100, max_depth=3, learning_rate=0.1,
        gamma=0, subsample=0.8, colsample_bytree=0.8),
    'Naive Bayes': GaussianNB()
}

def concat_seasons():
    """Concatenate seasons data into a single DataFrame."""
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        curr_dir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(
                f'{curr_dir}/MRegularSeasonDetailedResults_{season}_matchups_avg_10_w_rating.csv')
            all_seasons = pd.concat([all_seasons, season_df])
        except FileNotFoundError:
            pass
    return all_seasons

def fit_model_scalar(model_param, model_name, features, targets):
    """Fit a model with scalar features."""
    tscv = TimeSeriesSplit(n_splits=10)
    scaler = StandardScaler()
    accuracies = []
    iteration = 1
    for train_index, test_index in tscv.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        # Scale the features
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model_param.fit(x_train_scaled, y_train)

        y_pred = model_param.predict(x_test_scaled)
        current_accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} iteration {iteration} scalar accuracy: {current_accuracy:.5f}')
        accuracies.append(current_accuracy)
        iteration += 1

    print(f'{model_name} avg scalar accuracy: {np.mean(accuracies):.5f}')
    return model_param, accuracies

def time_series_split_by_season(model, model_name, seasons_data):
    """Splits the data into training and testing sets by season."""
    scaler = StandardScaler()
    accuracies = []
    print(f'Training {model_name} model...')
    for i in range(1, len(seasons_data)):
        print(f'Testing on Season {seasons_data[i]["Season"].unique()[0]}')
        train = pd.concat(seasons_data[:i])
        train = train.dropna(axis='columns', how='any')
        x_train = train.drop(
            ['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum',
             'team_1_Week_x', 'team_1_Week_y', 'team_2_TeamID', 'team_2_DayNum',
             'team_2_Week_x','team_2_Week_y', 'team_1_won'], axis=1)
        y_train = train['team_1_won']
        x_train = scaler.fit_transform(x_train)
        test = seasons_data[i]
        test = test.dropna(axis='columns', how='any')
        x_test = test.drop(
            ['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum',
             'team_1_Week_x', 'team_1_Week_y', 'team_2_TeamID', 'team_2_DayNum',
             'team_2_Week_x','team_2_Week_y', 'team_1_won'], axis=1)
        x_test = scaler.transform(x_test)
        y_test = test['team_1_won']
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy:.5f}')

    print(f'{model_name} avg scalar accuracy: {np.mean(accuracies):.5f}')
    return model, accuracies

def train_models(models):
    """Train multiple models."""
    seasons_df = concat_seasons()
    seasons_df = seasons_df.dropna(axis='columns', how='any')
    seasons = seasons_df['Season'].unique()
    seasons_data = [seasons_df[seasons_df['Season'] == season] for season in seasons]

    trained_models = {}
    accuracy_scores = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_model = {
            executor.submit(time_series_split_by_season, curr_model, curr_name, seasons_data): curr_name
            for curr_name, curr_model in models.items()
        }

        for future in as_completed(future_to_model):
            curr_name = future_to_model[future]
            try:
                model, accuracies = future.result()
                trained_models[curr_name] = model
                accuracy_scores[curr_name] = accuracies
            except Exception as exc:
                print(f'{curr_name} generated an exception: {exc}')

    return trained_models, accuracy_scores

if __name__ == '__main__':
    df = concat_seasons()
    df = df.dropna(axis='columns', how='any')
    X = df.drop(
        ['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum',
         'team_1_Week_x', 'team_2_TeamID', 'team_2_DayNum',
         'team_2_Week_x', 'team_1_won'], axis=1)
    y = df['team_1_won']

    trained_models, scores = train_models(classifiers)
    print(trained_models)
    scores = pd.DataFrame(scores)
    print(scores)
