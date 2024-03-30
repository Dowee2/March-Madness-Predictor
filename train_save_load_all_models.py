from joblib import dump, load

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd
import os

import predict_game_winner_by_avg as avg
import predict_game_winner_by_avg_10 as avg_10
import predict_game_winner_by_avg_w_ratings as avg_rating
import predict_game_winner_10_games_w_rating as rating

classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, learning_rate = 0.1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, activation = 'logistic', learning_rate = 'invscaling'),
        'Naive Bayes': GaussianNB()
    }


# Load the models
def load_models():
    """
    Loads all the trained models and returns them as a dictionary.
    
    Returns:
        models (dict): A dictionary containing all the trained models.
    """
    
    models = {}

    models['avg_10'] = load('models/avg_10_models.joblib')
    models['avg'] = load('models/avg_models.joblib')
    models['rating_rol10'] = load('models/rating_rol10_models.joblib')
    models['avg_rating'] = load('models/avg_rating_models.joblib')

    return models

# Save the models
def save_trained_models():
    """
    Trains all the models and saves them to the models folder.
    """

    avg_10_models, avg_10_models_accuracies = avg_10.train_models(classifiers)
    dump(avg_10_models, 'models/avg_10_models.joblib')
    pd.DataFrame(avg_10_models_accuracies).to_csv('models/avg_10_models_accuracies.csv')

    avg_models, avg_models_accuracies = avg.train_models(classifiers)
    dump(avg_models, 'models/avg_models.joblib')
    pd.DataFrame(avg_models_accuracies).to_csv('models/avg_models_accuracies.csv')

    rating_rol10_models, rating_rol10_models_accuracies = rating.train_models(classifiers)
    dump(rating_rol10_models, 'models/rating_rol10_models.joblib')
    pd.DataFrame(rating_rol10_models_accuracies).to_csv('models/rating_rol10_models_accuracies.csv')

    avg_rating_models, rating_rol10_models_accuracies = avg_rating.train_models(classifiers)
    dump(avg_rating_models, 'models/avg_rating_models.joblib')
    pd.DataFrame(rating_rol10_models_accuracies).to_csv('models/avg_rating_models_accuracies.csv')

if __name__ == '__main__':
    save_trained_models() 
    