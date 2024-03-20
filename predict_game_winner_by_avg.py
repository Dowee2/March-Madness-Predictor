import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
            season_df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_matchup.csv')
            all_seasons = pd.concat([all_seasons, season_df])          
        except:
            pass
    return all_seasons

def fit_model(model_param, model_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)

    # sfs = SequentialFeatureSelector(model_param, n_features_to_select=10)
    # sfs.fit(X_train, y_train)
    # X_train = sfs.transform(X_train)
    # X_test = sfs.transform(X_test)
    model_param.fit(X_train, y_train)
    y_pred = model_param.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} accuracy: {accuracy:.2f}')

def fit_model_scalar(model_param, model_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model
    model_param.fit(X_train_scaled, y_train)
    y_pred = model_param.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} scalar accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    df = pd.read_csv('data/Mens/Season/2015/MRegularSeasonDetailedResults_2015_matchup.csv')
    #df = concat_seasons()
    X = df.drop(['Season','DayNum','team_1', 'team_2', 'team_1_won'], axis=1)
    y = df['team_1_won']

    # param_grid = {
    # 'n_estimators': [50, 100, 200],
    # 'max_depth': [3, 6, 9],
    # 'learning_rate': [0.01, 0.1, 0.3],
    # 'subsample': [0.7, 0.8, 0.9],
    # 'colsample_bytree': [0.7, 0.8, 0.9],
    # 'gamma': [0, 0.1, 0.2]
    # }

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)
    # grid_search = GridSearchCV(XGBClassifier(random_state=3270), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # fitted = grid_search.fit(X_train, y_train)
    # print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    # y_pred = fitted.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Random Forest accuracy: {accuracy:.2f}')
    

    # Train the models and fit model
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs', ),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8)
    }

    for name, model in models.items():
        fit_model(model, name, X, y)
        fit_model_scalar(model, name, X, y)