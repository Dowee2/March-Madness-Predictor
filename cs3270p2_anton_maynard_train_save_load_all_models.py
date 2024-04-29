
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
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import pandas as pd

from trainers.regular_season import cs3270p2_anton_maynard_predict_game_winner_by_avg as avg
from trainers.regular_season import cs3270p2_anton_maynard_predict_game_winner_by_avg_5 as avg_5
from trainers.regular_season import cs3270p2_anton_maynard_predict_game_winner_by_avg_w_ratings as avg_rating
from trainers.regular_season import cs3270p2_anton_maynard_predict_game_winner_5_games_w_rating as rating
from trainers.tourney import cs3270p2_anton_maynard_predict_tourney_game_winner_by_avg as t_avg
from trainers.tourney import cs3270p2_anton_maynard_predict_tourney_game_winner_by_avg_5 as t_avg_5
from trainers.tourney import cs3270p2_anton_maynard_predict_tourney_game_winner_by_avg_w_ratings as t_avg_rating
from trainers.tourney import cs3270p2_anton_maynard_predict_tourney_game_winner_5_games_w_rating as t_rating


classifers_for_voting = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, 
                                                max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, 
                                                max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, 
                                                  penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, 
                                 learning_rate = 0.1, gamma = 0, subsample = 0.8,
                                 colsample_bytree = 0.8),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, 
                             activation = 'logistic', learning_rate = 'invscaling'),
        'Naive Bayes': GaussianNB()
}
classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=3270, 
                                                max_depth=10, min_samples_split=10),
        'Random Forest': RandomForestClassifier(random_state=3270, n_estimators=200, 
                                                max_depth=10, min_samples_split=10, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=3270, max_iter=1000, 
                                                  penalty = None, solver = 'lbfgs'),
        'XGBoost': XGBClassifier(random_state = 3270, n_estimators = 100, max_depth = 3, 
                                 learning_rate = 0.1, gamma = 0, subsample = 0.8,
                                 colsample_bytree = 0.8),
        'MLP': MLPClassifier(random_state = 3270, hidden_layer_sizes = (60,30,15), max_iter = 25, 
                             activation = 'logistic', learning_rate = 'invscaling'),
        'Naive Bayes': GaussianNB()
    }
classifiers['Voting'] = VotingClassifier(estimators=[(name, clf) for name, clf in classifers_for_voting.items()], n_jobs=-1, voting='hard')




def load_models():
    """
    Loads all the trained models and returns them as a dictionary.
    
    Returns:
        models (dict): A dictionary containing all the trained models.
    """

    models = {}
    models['avg_5'] = load('models/avg_5_models.joblib')
    models['avg'] = load('models/avg_models.joblib')
    models['rating_rol5'] = load('models/rating_rol5_models.joblib')
    models['avg_rating'] = load('models/avg_rating_models.joblib')

    models['tourney_avg'] = load('models/tourney_avg_models.joblib')
    models['tourney_avg_5'] = load('models/tourney_avg_5_models.joblib')
    models['tourney_avg_rating'] = load('models/tourney_avg_rating_models.joblib')
    models['tourney_rating_rol5'] = load('models/tourney_rating_rol5_models.joblib')

    return models


def save_trained_models():
    """
    Trains all the models and saves them to the models folder.
    """

    avg_5_models, avg_5_models_accuracies = avg_5.train_models(classifiers)
    dump(avg_5_models, 'models/avg_5_models.joblib')
    pd.DataFrame(avg_5_models_accuracies).to_csv('models/avg_5_models_accuracies.csv')

    avg_models, avg_models_accuracies = avg.train_models(classifiers)
    dump(avg_models, 'models/avg_models.joblib')
    pd.DataFrame(avg_models_accuracies).to_csv('models/avg_models_accuracies.csv')

    rating_rol5_models, rating_rol5_models_accuracies = rating.train_models(classifiers)
    dump(rating_rol5_models, 'models/rating_rol5_models.joblib')
    pd.DataFrame(rating_rol5_models_accuracies).to_csv('models/rating_rol5_models_accuracies.csv')

    avg_rating_models, avg_rating_models_accuracies = avg_rating.train_models(classifiers)
    dump(avg_rating_models, 'models/avg_rating_models.joblib')
    avg_rating_models_accuracies = pd.DataFrame(avg_rating_models_accuracies)
    pd.DataFrame(avg_rating_models_accuracies).to_csv('models/avg_rating_models_accuracies.csv')

    tourney_avg_models, tourney_avg_models_accuracies = t_avg.train_models(classifiers)
    dump(tourney_avg_models, 'models/tourney_avg_models.joblib')
    pd.DataFrame(tourney_avg_models_accuracies).to_csv('models/tourney_avg_models_accuracies.csv')

    tourney_avg_5_models, tourney_avg_5_models_accuracies = t_avg_5.train_models(classifiers)
    dump(tourney_avg_5_models, 'models/tourney_avg_5_models.joblib')
    pd.DataFrame(tourney_avg_5_models_accuracies).to_csv('models/tourney_avg_5_models_accuracies.csv')

    tourney_avg_rating_models, tourney_avg_rating_models_accuracies = t_avg_rating.train_models(classifiers)
    dump(tourney_avg_rating_models, 'models/tourney_avg_rating_models.joblib')
    pd.DataFrame(tourney_avg_rating_models_accuracies).to_csv('models/tourney_avg_rating_models_accuracies.csv')

    tourney_rating_rol5_models, tourney_rating_rol5_models_accuracies = t_rating.train_models(classifiers)
    dump(tourney_rating_rol5_models, 'models/tourney_rating_rol5_models.joblib')
    pd.DataFrame(tourney_rating_rol5_models_accuracies).to_csv('models/tourney_rating_rol5_models_accuracies.csv')

if __name__ == '__main__':
    save_trained_models()
