#Udacity Data Analyst Project 2-Investigating a Dataset
#Author: Manas Sarma
#Dataset Chosen: Titanic Data

import csv #required to read the csv
from pandas import Series, DataFrame #imports Series and DataFrames from the pandas library
import pandas as pd #extension for pandas
import numpy as np #imports numpy and gives it the extension np
import matplotlib.pyplot as plt #imports plotting library from Matlab and gives plots the extension plt

fields = ['Fare', 'Survived'] #stores the variables 'Fare' and 'Survived' to be plotted
fields1 = ['Pclass', 'Survived'] #stores the variables 'Pclass' and 'Survived' to be plotted
fields2 = ['Age', 'Fare'] #stores the variables 'Age' and 'Fare' to be plotted 
fields3 = ['Sex', 'Parch'] #stores the variables 'Sex' and 'Parch' to be plotted
fields4 = ['Sex', 'SibSp'] #stores the variables 'Sex' and 'Sibsp' to be plotted
fields5 = ['Sex', 'Survived'] #stores the variables 'Sex' and 'Survived' to be plotted

data_df = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields) #makes a data frame of fare and survival rates
data_df1 = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields1) #makes a data frame of cabin class and survival rates
data_df2 = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields2) #makes a data frame of age and fare
data_df3 = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields3) #makes a data frame of gender and accompanying parents/children
data_df4 = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields4) #makes a data frame of gender and accompanying siblings/spouses
data_df5 = pd.read_csv('/Users/mnsarma/Data_Analysis/titanic_data.csv', skipinitialspace=True, usecols=fields5) #makes a data frame of gender and survival

print data_df.Fare #prints the contents of the 'Fare' field in the data frame named data.df
print data_df.Survived #prints the contents of the 'Survived' field in the data frame named data.df
print data_df1.Pclass #prints the contents of the 'Pclass' field in the data frame data.df1
print data_df2.replace(r'\s+', np.nan, regex=True) #replaces blank values with NaN
print data_df3.replace(r'\s+', np.nan, regex=True) #replaces blank values with NaN
print(data_df4.Sex) #Prints the gender of the passengers
print(data_df4.SibSp) #Prints the number of siblings or spouses accompanying the passengers

################################################################################

#this for-loop prints out the number of people who traveled in first class

count = 0
for i in data_df1.Pclass:
    if i == 1:
        count += 1
print count #number of people traveling in first class

#this for-loop prints out the number of people who traveled in second-class

count1 = 0
for i in data_df1.Pclass:
    if i == 2:
        count1 += 1
print count1 #number of people traveling in second class

#this for-loop prints out the number of people who traveled in third-class

count2 = 0
for i in data_df1.Pclass:
    if i == 3:
        count2 += 1
print count2 #number of people traveling in third class

#Pie chart displaying the percentage of people in each cabin class

#labels = 'First', 'Second', 'Third'
#sizes = [count, count1, count2]
#colors = ['gold', 'green', 'coral']
#explode = (0.1, 0, 0)
#plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
#plt.axis('equal')
#plt.show()

################################################################################

#Plot for Fare rates versus Survival rates

#data_df.boxplot('Fare', 'Survived')
#plt.xticks([1, 2], ['Died', 'Survived'])
#plt.show()

################################################################################

#Plot for Passenger class versus Survival rates

#data_df1['Survival'] = data_df1.Survived.map({0 : 'Died', 1 : 'Survived'})
#df = data_df1.groupby('Pclass')['Survival'].value_counts().unstack()
#df.plot(kind = 'bar')
#plt.title('Pclass/Survival')
#plt.xlabel('Pclass')
#plt.ylabel('Survived')
#plt.show()

################################################################################

#Plot for Age versus Fare

#colors = 'red'
#c = colors
#area = np.pi * (15 * np.random.rand(50))**2
#s = area
#plt.xlabel('Age') #sets the label 'Age' for the x-axis
#plt.ylabel('Fare') #sets the label 'Fare' for the x-axis
#plt.scatter(data_df2.Age, data_df2.Fare, s, c, alpha=0.5)
#plt.show()

################################################################################

#Plot for Sex versus Parents/Children

#df = data_df3.groupby('Sex')['Parch'].value_counts().unstack()
#df.plot(kind = 'bar')
#plt.title('Sex/Parch')
#plt.xlabel('Sex')
#plt.ylabel('Parch')
#plt.show()

################################################################################

#Plot for Sex versus Siblings/Spouses

#df = data_df4.groupby('Sex')['SibSp'].value_counts().unstack()
#df.plot(kind = 'bar')
#plt.title('Sex/SibSp')
#plt.xlabel('Sex')
#plt.ylabel('SibSp')
#plt.show()

################################################################################

#Plot for Sex versus Survival

#data_df5['Survival'] = data_df5.Survived.map({0 : 'Died', 1 : 'Survived'})
#df = data_df5.groupby('Sex')['Survival'].value_counts().unstack()
#df.plot(kind = 'bar')
#plt.title('Sex/Survival')
#plt.xlabel('Sex')
#plt.ylabel('Survived')
#plt.show()

################################################################################