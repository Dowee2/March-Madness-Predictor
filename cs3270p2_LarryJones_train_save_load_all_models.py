
#pylint: disable=C0114 ,C0301,E0401
"""
This script trains and saves multiple machine learning models for predicting the outcome of March Madness games.

The script defines several classifiers including Decision Tree, Random Forest, Logistic Regression, XGBoost, MLP, and Naive Bayes.

Functions:
    - load_models(): Loads all the trained models and returns them as a dictionary.
    - save_trained_models(): Trains all the models and saves them to the models folder.
"""
from joblib import dump, load

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd

import cs3270p2_LarryJones_predict_game_winner_by_avg as avg
import cs3270p2_LarryJones_predict_game_winner_by_avg_10 as avg_10
import cs3270p2_LarryJones_predict_game_winner_by_avg_w_ratings as avg_rating
import cs3270p2_LarryJones_game_winner_10_games_w_rating as rating

classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, 
                                                max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=300, 
                                                max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=2000, 
                                                  penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 300, max_depth = 3, 
                                 learning_rate = 0.1, gamma = 0, subsample = 0.8,
                                 colsample_bytree = 0.8, tree_method='hist', device = 'cuda'),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, 
                             activation = 'logistic', learning_rate = 'invscaling'),
        'Naive Bayes': GaussianNB()
    }


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

    avg_rating_models, avg_rating_models_accuracies = avg_rating.train_models(classifiers)
    dump(avg_rating_models, 'models/avg_rating_models.joblib')
    avg_rating_models_accuracies = pd.DataFrame(avg_rating_models_accuracies)
    pd.DataFrame(avg_rating_models_accuracies).to_csv('models/avg_rating_models_accuracies.csv')


if __name__ == '__main__':
    save_trained_models()