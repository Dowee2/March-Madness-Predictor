NCAA Tournament Prediction Model
Overview
This project aims to predict the outcomes of NCAA tournament games using various machine learning models trained on historical data. The models consider different datasets representing team performances across seasons, incorporating both statistical averages and ratings.

How to Use
Main Scripts
---------------
train_save_load_all_models.py: 
Central script for training, saving, and loading models. 
Trains models on designated data variants and saves them along with their accuracy metrics in the /models directory.
This script is your starting point for model training.

predict_tourney_bracket.py: 
Predicts the outcome of the specified season's NCAA tournament games using all models trained by train_save_load_all_models.py. 
Outputs the completed tournament bracket based on these predictions.

Prediction and Training Scripts
-------------------------------
predict_game_winner_10_games_w_rating.py: 
    Trains models on the 10_games_w_rating dataset and predicts the winners for the specified season games. 
    This dataset focuses on the last 10 games' statistics coupled with current team ratings.

predict_games_winner_by_avg_w_rating.py: 
    Trains models on the avg_w_rating dataset and predicts the winners for the specified season games. 
    This variant averages season-long team performances and includes ratings.

predict_games_winner_by_avg_10.py: 
    Trains models on the avg_10 dataset and predicts the winners for the specified season games. 
    It averages the last 10 games' statistics for each team.

predict_game_winner_by_avg_stats.py: 
    Trains models on the avg_stats dataset and predicts the winners for the specified season games. 
    This dataset summarizes season-long team statistics.

Data Preprocessing Scripts
---------------------------
prep_avg_stats_10_games.py: 
Creates the avg_10 dataset by calculating average statistics from each team's last 10 games for that specific matchup, setting the stage for model training.

prep_avg_stats_w_rating.py: 
Creates the avg_w_rating dataset, averaging each team's seasonal performances and incorporating team ratings, enriching the data for subsequent model training.

prep_avg_stats_match.py: 
Creates the  avg_stats dataset, summarizing seasonal averages of all game statistics per team.

prep_avg_10_games_w_rating.py: 
Creates the 10_games_w_rating dataset by averaging statistics from the last 10 games and including each team's rating specific matchup.

Recommendations
Initial Setup: 
Begin by running train_save_load_all_models.py to train, save, and evaluate models. To focus on specific models, selectively comment out others within the script.

Tournament Predictions: 
After training, use predict_tourney_bracket.py for comprehensive tournament outcome predictions.
A 2024_Bracket_Actuals_v_Preds.xlsx file has already been complied for the 2024 season, comparing actual outcomes to model predictions. 
This can be found in the main directory. It was last updated on 3/30/2024.

Seasonal Predictions: 
For predictions specific to the 2024 season using various data variants, employ the respective prediction scripts. 
Ensure data preprocessing scripts are executed beforehand to update or create datasets as necessary.

