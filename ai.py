import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


df = pd.read_csv('Depth.csv') #Reading the CSV file containing ocean depth and magnetic field data


'''Preparing the data for a RandomForest regression algorithm by splitting the dataset into training and testing sets. '''
dataset = df[df['OceanDepth'] < -60] #Only including data for depths less than -60
x = dataset.iloc[:, 3:].values # select the magnetic field components as the input variables
y = dataset.iloc[:, 2].values # select the ocean depth as the target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

'''Train a RandomForestRegressor model on the training data and predict the ocean depth for the test data'''

regressor = RandomForestRegressor(n_estimators=30, random_state=23) # create a model with 30 decision trees
regressor.fit(x_train, y_train) # fit the model to the training data
y_predicted = regressor.predict(x_test) # predict the ocean depth for the test data


'''calculate the R2 score to evaluate the accuracy of the model predictions'''

score = r2_score(y_test, y_predicted)
print(f"R2 score: {score*100:.2f}%") #Printing the R2 score as a percentage



''' In conclusion, this code performs a data analysis on ocean depth and magnetic field data,
 exploring the relationship between these variables using scatter plots 
 and building a predictive model using a RandomForest regression algorithm. 
 The resultant magnetic field is calculated, and a box plot is used to visualize the distribution of the magnetic field values. 
 The R2 score is then calculated to evaluate the accuracy of the model predictions. 
 While the purpose of the analysis is not explicitly stated, it may be useful in fields such as geology 
 and oceanography that seek to understand the structure and composition of the Earth's crust.    '''