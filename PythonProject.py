# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 22:51:10 2021

@author: mdsou
"""

import plotly.figure_factory as ff
import warnings
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

# for basic mathematics operation
import numpy as np
from pandas import plotting

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)

warnings.filterwarnings('ignore')

#Load and read data
df=pd.read_csv('C:\\Users\\mdsou\\Downloads\\heart.csv')

pd.set_option('display.max_columns', None)
#Describe basic data attributes
py.iplot(ff.create_table(df.head()))
df.head()
df.describe()
df.isnull().any()
df.corr()
#Vizualize some of the variables
sns.pairplot(df)
plt.title('Pairplot for the Data', fontsize = 16)
plt.show()

sns.histplot(df['MaxHR'])
sns.histplot(df['Oldpeak'])
sns.histplot(df['HeartDisease'])
sns.histplot(df['Cholesterol'])
sns.histplot(df['RestingBP'])
sns.histplot(df['Age'], color = 'red')

sns.heatmap(df.corr(),annot=True,lw=1)


plt.rcParams['figure.figsize'] = (18, 18)

plt.subplot(2, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['MaxHR'])
plt.title('Heart Rate of patients', fontsize = 20)
plt.xlabel('Maximum Heart Rate')
plt.ylabel('Count')


plt.subplot(2, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['Age'], color = 'red')
plt.title('Distribution of Age', fontsize = 20)
plt.xlabel('Range of Ages')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['Cholesterol'], color = 'green')
plt.title('Distribution of Cholesterol levels in Patients', fontsize = 20)
plt.xlabel('Cholesterol levels')
plt.ylabel('Count')
plt.show()


plt.rcParams['figure.figsize'] = (18, 7)
sns.violinplot(df['Sex'], df['HeartDisease'], palette = 'rainbow',
               split = 'bool')
plt.title('Sex versus Heart Disease', fontsize = 20)
plt.show()

plt.rcParams['figure.figsize'] = (36, 14)
sns.violinplot(df['RestingECG'], df['HeartDisease'], palette = 'rainbow')
plt.title('ECG versus Heart Disease', fontsize = 20)
plt.show()

plt.rcParams['figure.figsize'] = (22, 11)
sns.violinplot(df['ExerciseAngina'], df['HeartDisease'], palette = 'rainbow')
plt.title('Angina versus Heart Disease', fontsize = 20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)
sns.violinplot(df['ST_Slope'], df['HeartDisease'], palette = 'rainbow')
plt.title('ST versus Heart Disease', fontsize = 20)
plt.show()

print(df['Cholesterol'])
df.loc[df['Cholesterol'] == 0 ]
print(df[df.Cholesterol != 0])
#df.drop('Cholesterol')
df_filtered = df[df['Cholesterol'] >= 10]
df_filtered.head(10)
df.drop(df[df['Cholesterol'] == 0].index, inplace = True)
df.describe()

#Create copy of the dataframe
scaled_features = df.copy()

#Don't include the unwanted columns in the transformation:
col_names = ['Cholesterol', 'MaxHR', 'RestingBP']
features = df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
df = df.drop(['RestingECG', 'RestingBP', 'MaxHR'], axis = 1)

#Create dummy variables for categorical data columns
categorical_columns = ['ST_Slope', 'ExerciseAngina', 'ChestPainType', 'Sex']
df2 = pd.get_dummies(data = df,
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
df2[col_names] = features
print(df2.head())

#print featuees of data
x = df2.drop(['HeartDisease'], axis = 1)
y = df2['HeartDisease']
#Add constant for Intercept
x = sm.add_constant(x)
#Divide my data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 5)
#Check the size of each set
print(x_train.columns)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x = pd.DataFrame(x)
y = pd.DataFrame(y)

plt.rcParams['figure.figsize'] = [16,10]
fig, axs= plt.subplots(1,1)
sns.histplot(y,bins=10,color='r',kde=True, stat = 'percent')
axs[0].set_title('Percentage with Heart Disease')

# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0)
# Train model
logisticModel = logistic_regression.fit(x_train, y_train)
print("Accuracy: ", logisticModel.score(x_train, y_train))
print("Intercept: ", logisticModel.intercept_)
print("Coefficients: ", logisticModel.coef_)
logisticModel.score(x_train, y_train)
confusion_matrix(y, logisticModel.predict(x))
print(classification_report(y, logisticModel.predict(x)))

y_pred=logisticModel.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

model = sm.Logit(y, x)
result = model.fit(method='newton')
result.summary()