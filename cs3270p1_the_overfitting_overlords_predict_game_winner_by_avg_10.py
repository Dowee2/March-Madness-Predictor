#!/usr/bin/env python3

# pylint: disable=W0718,W0621,E0401,C0301,R0914,C0103

"""
This module is used for training various machine learning models on NCAA Men's Basketball data to predict game outcomes.
It demonstrates the use of time series split for model evaluation and uses multiple classifiers including Decision Tree,
Random Forest, Logistic Regression, xGBoost, MLP, and Naive Bayes.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

def concat_seasons():
    """
    Concatenate all season data files into a single DataFrame.

    Returns:
        all_seasons (DataFrame): DataFrame containing all season data.
    """
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg_10.csv')
            all_seasons = pd.concat([all_seasons, season_df])
        except FileNotFoundError:
            pass
    return all_seasons

def fit_model_scalar(model_param, model_name, x, y):
    """
    Fit a model with scaled features and print its accuracy for each iteration.

    Parameters:
        model_param (model): The model to be trained.
        model_name (str): The name of the model.
        x (DataFrame): The input features.
        y (Series): The target variable.
    """
    tscv = TimeSeriesSplit(n_splits=20)
    scaler = StandardScaler()
    accuracies = []
    iteration = 1
    for train_index, test_index in tscv.split(x):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model_param.fit(x_train_scaled, y_train)
        y_pred = model_param.predict(x_test_scaled)
        current_accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} iteration {iteration} scalar accuracy: {current_accuracy:.5f}')
        accuracies.append(current_accuracy)
        iteration += 1
    avg_acc = np.mean(accuracies)
    print(f'{model_name} avg scalar accuracy: {avg_acc:.5f}')

def timeseriessplit_by_season(model, model_name, seasons_data):
    """
    Splits the data into training and testing sets by season. Model is trained on all data up to a certain season and tested on the next season until the last season.

    Parameters:
    - model (model): The model to be trained and tested.
    - model_name (str): The name of the model.
    - seasons_data (list): A list of DataFrames containing data for each season.

    Returns:
    - tuple: The trained model and a list of accuracy scores for each season.
    """
    scaler = StandardScaler()
    accuracies = []
    print(f'Training {model_name} model...')
    for i in range(1, len(seasons_data)):
        print(f'Testing on Season {seasons_data[i]["Season"].unique()[0]}')
        train = pd.concat(seasons_data[:i])
        train = train.dropna(axis='columns', how='any')
        x_train = train.drop(['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum', 'team_1_Week', 'team_2_TeamID', 'team_2_DayNum', 'team_2_Week', 'team_1_won'], axis=1)
        y_train = train['team_1_won']
        x_train = scaler.fit_transform(x_train)
        test = seasons_data[i]
        test = test.dropna(axis='columns', how='any')
        x_test = test.drop(['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum', 'team_1_Week', 'team_2_TeamID', 'team_2_DayNum', 'team_2_Week', 'team_1_won'], axis=1)
        x_test = scaler.transform(x_test)
        y_test = test['team_1_won']
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'accuracy: {accuracy:.5f}')

    print(f'{model_name} avg scalar accuracy: {np.mean(accuracies):.5f}')
    return model, accuracies

def train_models(models):
    """
    Train multiple models using time series split by season and return their accuracies.

    Parameters:
        models (dict): Dictionary of models to be trained with their names as keys.

    Returns:
        tuple: Dictionary of trained models and dictionary of their accuracy scores.
    """
    seasons_df = concat_seasons()
    seasons = seasons_df['Season'].unique()
    seasons_data = [seasons_df[seasons_df['Season'] == season] for season in seasons]

    trained_models = {}
    accuracy_scores = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_model = {executor.submit(timeseriessplit_by_season, curr_model, curr_name, seasons_data): curr_name for curr_name, curr_model in models.items()}
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
    df = pd.read_csv('data/Mens/Season/2024/MRegularSeasonDetailedResults_2024_matchups_avg_10.csv')
    df = concat_seasons()
    df = df.dropna(axis='columns', how='any')
    x = df.drop(['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum', 'team_1_Week', 'team_2_TeamID', 'team_2_DayNum', 'team_2_Week', 'team_1_won'], axis=1)
    y = df['team_1_won']
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty=None, solver='lbfgs'),
        'xGBoost': XGBClassifier(random_state=3270, n_estimators=100, max_depth=3, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.8),
        'MLP': MLPClassifier(random_state=3270, hidden_layer_sizes=(60, 30, 15), max_iter=25, activation='logistic', learning_rate='invscaling'),
        'Naive Bayes': GaussianNB()
    }
    for name, classifier in classifiers.items():
        fit_model_scalar(classifier, name, x, y)
