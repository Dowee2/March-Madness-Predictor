import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Read the dataset from the string
df = pd.read_csv('data/Mens/Season/2015/MRegularSeasonDetailedResults_2015_matchup.csv')

# Feature Engineering: Creating differences in key metrics
features_df = df.drop(columns=['Season', 'DayNum', 'team_1', 'team_2', 'team_1_won'])

X = features_df
y = df['team_1_won']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3072)


# Training the Logistic Regression Classifier
log_reg_clr = LogisticRegression(random_state=3072)
log_reg_clr.fit(X_train, y_train)
# Predicting and evaluating the model
predictions = log_reg_clr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Logistic Regression Predictions: {predictions}")
print(f"Logistic Regression Accuracy: {accuracy}")



# Training the Random Forest Classifier
forest_clr = RandomForestClassifier(random_state=3072)
forest_clr.fit(X_train, y_train)
# Predicting and evaluating the model
predictions = forest_clr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Predictions: {predictions}")
print(f"Random Forest Accuracy: {accuracy}")




# Training the Decision Tree Classifier
tree_clr = DecisionTreeClassifier(random_state=3072)
tree_clr.fit(X_train, y_train)
# Predicting and evaluating the model
predictions = tree_clr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Decision Tree Predictions: {predictions}")
print(f"Decision Tree Accuracy: {accuracy}")