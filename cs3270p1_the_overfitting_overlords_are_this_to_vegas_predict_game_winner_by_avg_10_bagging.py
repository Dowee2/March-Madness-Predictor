#!/usr/bin/env python3

#pylint: disable=W0718,W0621,E0401,C0301,R0914

"""Module for training and evaluating classifiers using bagging and voting techniques."""

import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import numpy as np

def concat_seasons():
    """Concatenate seasons data into a single DataFrame."""
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        curr_dir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(f'{curr_dir}/MRegularSeasonDetailedResults_{season}_matchups_avg_10.csv')
            all_seasons = pd.concat([all_seasons, season_df])
        except FileNotFoundError:
            pass
    return all_seasons

def fit_model(model_param, model_name, features, targets):
    """Fit a single model and print its average accuracy."""
    tscv = TimeSeriesSplit(n_splits=10)
    accuracies = []
    for train_index, test_index in tscv.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
        model_param.fit(x_train, y_train)
        y_pred = model_param.predict(x_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_acc = np.mean(accuracies)
    print(f'{model_name} accuracy: {avg_acc:.2f}')

def fit_model_scalar(models, features, targets):
    """Fit multiple models using scalar features and a voting classifier."""
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    accuracies = []
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models[0]), ('lr', models[1]), ('xgb', models[2]), ('mlp', models[3])
        ],
        voting='hard'
    )
    for train_index, test_index in tscv.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]

        # Scale the features
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        voting_clf.fit(x_train_scaled, y_train)

        y_pred = voting_clf.predict(x_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_acc = np.mean(accuracies)
    print(f'Accuracy: {avg_acc:.2f}')

if __name__ == '__main__':
    df = pd.read_csv('data/Mens/Season/2022/MRegularSeasonDetailedResults_2022_matchups_avg_10.csv')
    df = df.dropna(axis='columns', how='any')
    features = df.drop(
        ['Season', 'DayNum', 'team_1_TeamID', 'team_1_DayNum',
         'team_1_Week', 'team_2_TeamID', 'team_2_DayNum', 'team_2_Week', 'team_1_won'],
        axis=1
    )
    targets = df['team_1_won']

    # Train the models and fit model
    classifiers = [
        RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10),
        LogisticRegression(random_state=3270, max_iter=1000, penalty=None, solver='lbfgs'),
        XGBClassifier(random_state=3270, n_estimators=100, max_depth=3, learning_rate=0.1, gamma=0, subsample=0.8, colsample_bytree=0.8),
        MLPClassifier(random_state=3270, hidden_layer_sizes=(60, 30, 15), max_iter=25, activation='logistic', learning_rate='invscaling')
    ]

    fit_model_scalar(classifiers, features, targets)
