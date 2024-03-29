import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np


# Read the dataset from the string


def concat_seasons():
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg_10.csv')
            all_seasons = pd.concat([all_seasons, season_df])          
        except:
            pass
    return all_seasons

def fit_model(model_param, model_name, X, y):
    
    tscv = TimeSeriesSplit(n_splits=10)
    accuracies  = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model_param.fit(X_train, y_train)
        y_pred = model_param.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_acc = np.mean(accuracies)
    print(f'{model_name} accuracy: {avg_acc:.2f}')

def fit_model_scalar(models, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    accuracies  = []
    voting_clf = VotingClassifier(estimators=[('rf', models[0]), ('lr', models[1]), ('xgb', models[2]), ('mlp', models[3])], voting='hard')
    voting_clf.set_params(xgb = 'drop')
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        voting_clf.fit(X_train_scaled, y_train)
        
        y_pred = voting_clf.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_acc = np.mean(accuracies)
    print(f'accuracy: {avg_acc:.2f}')

if __name__ == '__main__':
    df = pd.read_csv('data/Mens/Season/2022/MRegularSeasonDetailedResults_2022_matchups_avg_10.csv')
    #df = concat_seasons()
    df = df.dropna(axis = 'columns', how= 'any')
    X = df.drop(['Season','DayNum','team_1_TeamID' ,'team_1_DayNum', 'team_1_Week','team_2_TeamID' ,'team_2_DayNum', 'team_2_Week', 'team_1_won'], axis=1)
    y = df['team_1_won']
    
    # Train the models and fit model
    classifiers = [
        RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10),
        LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs'),
        XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
        MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, activation = 'logistic', learning_rate = 'invscaling')
    ]

    fit_model_scalar(classifiers, X, y)