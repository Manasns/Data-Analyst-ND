#!/usr/bin/python

import sys
import pickle
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#I added the features 'deferral_payments,' 'deferred_income,' 'total_payments,'
#'bonus,' and 'loan_advances'

features_list = ['poi','salary', 'deferral_payments', 'deferred_income',
'total_payments', 'bonus', 'loan_advances'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

#the "data" variable stores the dictionary and the values for salary and deferral payments
data = featureFormat(data_dict, ['salary', 'deferral_payments'])

print "The maximum value is %d" % (data.max())

################################################################################

#plot showing the relationship between salary and deferral payments

# for entry in data:
#     salary = entry[0]
#     deferral_payments = entry[1]
#     plt.scatter(salary, deferral_payments)
# plt.xlabel("salary")
# plt.ylabel("deferral payments")
# plt.show()

################################################################################

#for-loops to determine outliers within salary and deferral payments

outlier = []

for key in data_dict:
    amt = data_dict[key]['salary']
    if amt == 'NaN':
        continue
    outlier.append((key, int(amt)))
#prints the list [('SKILLING JEFFREY K', 1111258), ('LAY KENNETH L', 1072321)]
print(sorted(outlier, key=lambda x:x[1], reverse=True)[:2])

for item in data_dict:
    amount = data_dict[item]['deferral_payments']
    if amount == 'NaN':
        continue
    outlier.append((item, int(amount)))
#prints the list [('FREVERT MARK A', 6426990), ('HORTON STANLEY C', 3131860)]
print(sorted(outlier, key=lambda y:y[1], reverse=True)[:2])

################################################################################

data_dict.pop('TOTAL', 0)
data1 = featureFormat(data_dict, ['deferred_income', 'total_payments'])

print "The maximum value is %d" % (data1.max())

#plot showing the relationship between deferred income and bonus

#for value in data1:
#    deferred_income = value[0]
#    total_payments = value[1]
#    plt.scatter(deferred_income, total_payments)
#plt.xlabel("deferred income")
#plt.ylabel("total_payments")
#plt.show()

################################################################################

#for-loops to determine outliers within deferred income and total payments

outlier1 = []

for key in data_dict:
    i = data_dict[key]['deferred_income']
    if i == 'NaN':
        continue
    outlier1.append((key, int(i)))
#prints the list [('BOWEN JR RAYMOND M', -833), ('GAHN ROBERT S', -1042)]
print(sorted(outlier1, key=lambda x:x[1], reverse=True)[:2])

for keys in data_dict:
    bon = data_dict[keys]['total_payments']
    if bon == 'NaN':
        continue
    outlier1.append((keys, int(bon)))
#prints the list [('LAY KENNETH L', 103559793), ('FREVERT MARK A', 17252530)]
print(sorted(outlier1, key=lambda y:y[1], reverse=True)[:2])

#data_dict.pop('TOTAL', 0)

################################################################################

data2 = featureFormat(data_dict, ['bonus', 'loan_advances'])

#plot showing the relationship between bonus and loan advances

data_dict.pop('TOTAL', 0)
print "The maximum value is %d" % data2.max()

#for quant in data2:
#    bonus = quant[0]
#    loan_advances = quant[1]
#    plt.scatter(bonus, loan_advances)
#plt.xlabel("bonus")
#plt.ylabel("loan advances")
#plt.show()

################################################################################

#for-loops to determine outliers within bonus and loan advances

outlier2 = []

for unit in data_dict:
    i = data_dict[unit]['bonus']
    if i == 'NaN':
        continue
    outlier2.append((unit, int(i)))
#prints the list [('LAVORATO JOHN J', 8000000), ('LAY KENNETH L', 7000000)]
print(sorted(outlier2, key=lambda x:x[1], reverse=True)[:2])

for units in data_dict:
    bon = data_dict[units]['loan_advances']
    if bon == 'NaN':
        continue
    outlier2.append((units, int(bon)))
#prints the list [('LAY KENNETH L', 81525000), ('LAVORATO JOHN J', 8000000)]
print(sorted(outlier2, key=lambda y:y[1], reverse=True)[:2])

#data_dict.pop('TOTAL', 0)
data2 = featureFormat(data_dict, ['bonus', 'loan_advances'])

my_df = pd.DataFrame(data_dict).transpose() # transpose to get features as columns

# replace NaN with numpy nan so that we can use the .isnull() method
my_df= my_df.replace('NaN', np.nan)

# count up null values
print my_df.isnull().sum()

################################################################################

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#adds the features total_stock_value and expenses to the list

for keys in my_dataset:
    data_value = my_dataset[keys]
    total_stock_value = data_value['total_stock_value']
    expenses = data_value['expenses']
my_feature_list = features_list + ['total_stock_value', 'expenses']

#I created a new feature, profit, which equals salary - expenses

for employee, features in data_dict.iteritems():
    if features['salary'] == 'NaN' or features['expenses'] == 'NaN':
        features['profit'] = 'NaN'
    else:
        features['profit'] = features['salary'] - features['expenses']
print features['profit']

my_feature_list = features_list + ['total_stock_value', 'expenses'] + ['profit']

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, my_features_list, sort_keys = True)
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

k = model_selection.KFold(n_splits=10, random_state=42)
#shuffle=StratifiedShuffleSplit(labels_train, n_iter=1, test_size=0.3, random_state=20)

mod = LogisticRegression()
rec = RFE(mod, 5)
rec = rec.fit(features_train, labels_train)
#rec = rec.fit(features_train, labels_train)
print(rec.support_)
print(rec.ranking_)

#Creates a Gaussian naive Bayes classifier

# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# mat=confusion_matrix(labels_test, pred)
# reports = classification_report(labels_test, pred)
# results=model_selection.cross_val_score(clf, features_train, labels_train, cv=k)
# print accuracy_score(labels_test, pred, normalize=True)
# print reports
# print results.mean()
# print results.std()
# print pred
# print mat

################################################################################
#creates a decision tree classifier

decision = DecisionTreeClassifier(random_state=20)
sel = SelectKBest(k='all')
entries = [('feature_selection', sel), ('decision', decision)]
clf = Pipeline(entries)
clf.fit(features_train, labels_train)

param1 = {"decision__criterion": ["gini", "entropy"],
"decision__min_samples_split": [2, 5, 10],
"decision__max_depth": [None, 1, 2, 3, 4, 5],
"decision__min_samples_leaf": [1, 5, 10],
"decision__min_weight_fraction_leaf": [0, 0.25, 0.35, 0.45, 0.5],
"decision__max_leaf_nodes": [None, 5, 10, 15, 20],
"decision__min_impurity_split": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1, 1]}
scoring='f1'
verbose=10

guess = clf.predict(features_test)
mat1=confusion_matrix(labels_test, guess)
report = classification_report(labels_test, guess)
result = model_selection.cross_val_score(clf, features_train, labels_train, cv=k)
print accuracy_score(labels_test, guess, normalize=True)
print report
print result.mean()
print result.std()
print mat1

################################################################################
#creates a logistic regression classifier

#k = model_selection.KFold(n_splits=10, random_state=42)
#features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)

# clf = LogisticRegression()
# clf.fit(features_train, labels_train)
# 
# param2 = {'clf__C': [0.1, 1, 10], 'clf__n_jobs': [0, 0.0001, 0.001, 0.01, 0.1, 1],
# 'clf__max_iter': [1, 5, 10, 20, 50, 100]}
# scoring='f1'
# verbose=10
# 
# prediction = clf.predict(features_test)
# mat2=confusion_matrix(labels_test, prediction)
# rep = classification_report(labels_test, prediction)
# res = model_selection.cross_val_score(clf, features_train, labels_train, cv=k)
# print accuracy_score(labels_test, prediction, normalize=True)
# print(rep)
# print res
# print res.mean()
# print res.std()
# print(mat2)

estimator=clf

################################################################################
#GridSearchCV--to use for decision tree classifier

# grid1=GridSearchCV(estimator, param1, scoring)
# grid1.fit(features_train, labels_train)
# print ('The best score for the decision tree:', clf.best_score_)
# print clf

################################################################################

#GridSearchCV--to use for logistic regression classifier

# grid2=GridSearchCV(estimator, param2, scoring, cv=k)
# grid2.fit(features_train, labels_train)
# print ('The best score for the logistic regression classifier:', grid2.best_score_)
# print clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)