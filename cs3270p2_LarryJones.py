#!/usr/bin/env python3

#pylint: disable=W0718,W0621,E0401,C0301,R0914,C0103,C0412

"""
    This script trains a Long Short-Term Memory (LSTM) neural network to predict the next year's statistics for each team in the
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set seed for reproducibility
np.random.seed(3270)
tf.random.set_seed(3270)

def concat_seasons():
    """
    Concatenates all seasons' data into a single DataFrame.
    Returns:
    - DataFrame: A DataFrame containing data from all seasons.
    """
    data_location = 'data/Mens/Season/'
    seasons = os.listdir(data_location)
    all_seasons = pd.DataFrame()
    for season in seasons:
        currdir = os.path.join(data_location, season)
        try:
            season_df = pd.read_csv(f'{currdir}/MRegularSeasonDetailedResults_{season}_avg.csv')
            all_seasons = pd.concat([all_seasons, season_df])
        except FileNotFoundError:
            pass
    return all_seasons

def create_sequences(data, sequence_length):
    """
    Creates sequences from the data for each team separately.
    Parameters:
    - data (DataFrame): The scaled data.
    - sequence_length (int): The number of timesteps per sequence.
    Returns:
    - X, y (list of np.array): Sequences for training/testing.
    """
    X, y = [], []
    teams = data['TeamID'].unique()
    for team in teams:
        team_data = data[data['TeamID'] == team]
        for i in range(len(team_data) - sequence_length):
            X.append(team_data[i:(i + sequence_length)].drop(['TeamID'], axis=1))
            y.append(team_data.iloc[i + sequence_length].drop(['TeamID']))
    return np.array(X), np.array(y)

data = concat_seasons()
data = data.sort_values(by=['TeamID'])
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(['TeamID'], axis=1))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[1:], index=data.index)
data_scaled['TeamID'] = data['TeamID']

sequence_length = 4
X, y = create_sequences(data_scaled, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(np.array(X.tolist()), np.array(y.tolist()), test_size=0.2, random_state=3270)

model = Sequential([
    Input(shape=(sequence_length, X_train.shape[2])),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(20, return_sequences=False),
    Dense(y_train.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=callbacks)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

predicted_stats_all_teams = {}
for team_id in data['TeamID'].unique():
    team_data = data_scaled[data_scaled['TeamID'] == team_id]
    if len(team_data) > sequence_length:
        last_sequence = np.expand_dims(team_data.iloc[-sequence_length:].drop(['TeamID'], axis=1), axis=0)
        predicted_stats = model.predict(last_sequence)
        predicted_stats_all_teams[team_id] = scaler.inverse_transform(predicted_stats)[0]
for team_id, stats in predicted_stats_all_teams.items():
    print(f"Team {team_id} predicted next year's stats: {stats}")
