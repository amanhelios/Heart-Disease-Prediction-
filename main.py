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

