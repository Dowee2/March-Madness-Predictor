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
    Creates sequences from the data.

    Parameters:
    - data (np.array): The scaled data.
    - sequence_length (int): The number of timesteps per sequence.

    Returns:
    - X, y (np.array): Sequences for training/testing.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Load and prepare data
data = concat_seasons()
data = data.sort_values(by=['TeamID'])  
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(['TeamID'], axis=1))

sequence_length = 4  
X, y = create_sequences(data_scaled, sequence_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)

# Build the LSTM model
model = Sequential([
    Input(shape=(sequence_length, X_train.shape[2])),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(20, return_sequences=False),
    Dense(y_train.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=callbacks)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')

# Optionally, predict the next year's stats
last_sequence = np.expand_dims(data_scaled[-sequence_length:], axis=0)
next_year_prediction = model.predict(last_sequence)
predicted_stats = scaler.inverse_transform(next_year_prediction)
print("Predicted next year's stats:", predicted_stats)
