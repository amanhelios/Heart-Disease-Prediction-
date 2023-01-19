#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv data to a pandas DataFrame
heart_data = pd.read_csv('heart.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and colomns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# cheking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable 
heart_data['target'].value_counts()

#spliting the Features and Target
X = heart_data.drop(columns= 'target', axis=1)
Y = heart_data['target']

print(X)

#splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Model Training

#LOGISTIC REGRESSION
model = LogisticRegression()

# Training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
