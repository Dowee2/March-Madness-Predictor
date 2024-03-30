
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from concurrent.futures import  ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import os

# Read the dataset from the string


def concat_seasons():
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_matchups_avg.csv')
            all_seasons = pd.concat([all_seasons, season_df])          
        except:
            pass
    return all_seasons

def fit_model_scalar(model_param, model_name, X, y):
    tscv = TimeSeriesSplit(n_splits=20)
    scaler = StandardScaler()
    accuracies  = []
    iteration = 1
    for train_index, test_index in tscv.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_param.fit(X_train_scaled, y_train)
        
        y_pred = model_param.predict(X_test_scaled)
        cuurent_accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} iteration {iteration} scalar accuracy: {cuurent_accuracy:.5f}')
        accuracies.append(cuurent_accuracy)
        iteration += 1
    avg_acc = np.mean(accuracies)
    print(f'{model_name} avg scalar accuracy: {avg_acc:.5f}')

def TimeSeriesSplit_by_season(model, model_name, seasons_data):
    """
    Splits the data into training and testing sets by season. Model is trained on all data up to a certain season and tested on the next season until the last season. 

    Parameters:
    - seasons_data (list): A list of DataFrames containing data for each season.

    Returns:
    - list: A list of tuples containing training and testing sets for each season.
    """
    scaler = StandardScaler()
    accuracies = []
    print(f'Training {model_name} model...')
    for i in range(1, len(seasons_data)):
        print(f'Testing on Season {seasons_data[i]["Season"].unique()[0]}')
        train = pd.concat(seasons_data[:i])
        X_train = train.drop(['Season','DayNum','team_1_won'], axis=1)
        y_train = train['team_1_won']
        X_train = scaler.fit_transform(X_train)
        test = seasons_data[i]
        X_test = test.drop(['Season','DayNum','team_1_won'], axis=1)
        X_test = scaler.transform(X_test)
        y_test = test['team_1_won']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f'accuracy: {accuracy:.5f}')

    print(f'{model_name} avg scalar accuracy: {np.mean(accuracies):.5f}')
    return model, accuracies


def train_models(models):
    seasons_df = concat_seasons()
    seasons = seasons_df['Season'].unique()
    seasons_data = [seasons_df[seasons_df['Season'] == season] for season in seasons]

    trained_models = {}
    accuracy_scores = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit tasks
        future_to_model = {executor.submit(TimeSeriesSplit_by_season, curr_model, curr_name,  seasons_data): curr_name for curr_name, curr_model in models.items()}
        
        # Process as each task completes
        for future in as_completed(future_to_model):
            curr_name = future_to_model[future]
            try:
                model, accuracies = future.result()
                trained_models[curr_name] = model
                accuracy_scores[curr_name] = accuracies
            except Exception as exc:
                print(f'{curr_name} generated an exception: {exc}')

    # for curr_name, curr_model in models.items():
    #   model, accuracies = TimeSeriesSplit_by_season(curr_model, curr_name, seasons_data)
    #   trained_models[curr_name] = model
    #   accuracy_scores[curr_name] = accuracies
    return trained_models, accuracy_scores

if __name__ == '__main__':
    #df = pd.read_csv('data/Mens/Season/2015/MRegularSeasonDetailedResults_2015_matchups_avg.csv')
    df = concat_seasons()
    X = df.drop(['Season','DayNum','team_1', 'team_2', 'team_1_won'], axis=1)
    y = df['team_1_won']

    # Train the models and fit model
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs', ),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
        'Naive Bayes': GaussianNB()
    }

    #train_models(classifiers)
    for name, model in classifiers.items():
        fit_model_scalar(model, name, X, y)