# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

salaryData = pd.read_csv('Salary_Data.csv')

salaryData.dropna(inplace=True)

features = salaryData.YearsExperience.values
label = salaryData.Salary.values

#converting into 2 dimentional array
features = features.reshape(30,1)
label = label.reshape(30,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(features,label)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5]]))