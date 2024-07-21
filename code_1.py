import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
wine_dataset = pd.read_csv('winequality-red.csv')
# print(wine_dataset.head())
# print(wine_dataset.shape)
# wine_dataset.isnull().sum()
# wine_dataset.describe()
# sns.catplot(x='quality', data = wine_dataset, kind = 'count')
# plot = plt.figure(figsize=(5,5))
# sns.barplot(x='quality', y = 'volatile acidity', data = wine_dataset)
# plot = plt.figure(figsize=(5,5))
# sns.barplot(x='quality', y = 'citric acid', data = wine_dataset)

# correlation = wine_dataset.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# Preprocessing data
X = wine_dataset.drop(columns='quality', axis=1)
y = wine_dataset['quality']

Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
# Prediction on Test Data
test_data_prediction = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, test_data_prediction)
print("Accuracy Score: ", accuracy)

# Building a Predictive System
input_data = (11.0,0.3,0.58,2.1,0.054000000000000006,7.0,19.0,0.998,3.31,0.88,10.5)
# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==1):
  print('Good Quality Wine')
else:
    print('Bad Quality Wine')