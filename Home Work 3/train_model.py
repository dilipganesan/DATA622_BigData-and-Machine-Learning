#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:57:12 2018

CUNY DATA622 Home Work 2


@author: dilipganesan

When this is called using python train_model.py in the command line, 
this will take in the training dataset csv, perform the necessary data cleaning 
and imputation, and fit a classification model to the dependent Y. 
There must be data check steps and clear commenting for each step inside the 
.py file. The output for running this file is the random forest model 
saved as a .pkl file in the local directory. 

"""


## import the needed functions.
import pull_data as pulldata
#import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from contextlib import redirect_stdout
import pickle

## clean data method
def clean_data(df):
    # Removing all the variable which are not contributing to our prediction.
    df.drop('Name', inplace=True, axis=1)
    df.drop('Ticket', inplace=True, axis=1)
    df.drop('PassengerId', inplace=True, axis=1)
    df.drop('Cabin', inplace=True, axis=1)
    
    #Making sure the categorical variable Sex is getting converted to numerical.
    df['Sex'] = df['Sex'].replace('male', 1)
    df['Sex'] = df['Sex'].replace('female', 0)
    
    return df

## impute data method.
def impute_data(df, target):
    #print(df.columns)
    if target == 'train':
        # We already know that for train data we have Embarked and Age has NA.
        df['Embarked'] = df['Embarked'].fillna('S')
        mean_age = df['Age'].mean()
        df['Age'].fillna(mean_age, inplace=True)
        # We have to change the Embarked Categorical Variable to Numeric.
        df['Embarked'] = df['Embarked'].replace('S', 0)
        df['Embarked'] = df['Embarked'].replace('C', 1)
        df['Embarked'] = df['Embarked'].replace('Q', 2)
    elif target == 'test':
        # We know that for test data we have Fare and Age has NA.
        # Substituting NAs with mean.
        mean_fare = df['Fare'].mean()
        df['Fare'].fillna(mean_fare, inplace=True)
        mean_age = df['Age'].mean()
        df['Age'].fillna(mean_age, inplace=True)
        # We have to change the Embarked Categorical Variable to Numeric.
        df['Embarked'] = df['Embarked'].replace('S', 0)
        df['Embarked'] = df['Embarked'].replace('C', 1)
        df['Embarked'] = df['Embarked'].replace('Q', 2)
    else:
        raise ValueError('Not the correct target in imputation')
        
    return df
           
        
def print_model_output(sample_test, prediction, score=None):
    con_mat = metrics.confusion_matrix(sample_test, prediction)
    classreport = metrics.classification_report(sample_test, prediction)
    print(prediction)
    print(con_mat)
    print(classreport)
    try:
        with open('model_stats.txt', 'w') as text_file:
            with redirect_stdout(text_file):
                print('Random Forrest Classifier Model:')
                print(prediction)
                print('\nConfusion Matrix:')
                print(con_mat)
                print('\nClassification Rport:')
                print(classreport)
    except: 
        raise
 
    
## Main Program Execution Starts.
def building_model():
    ## Step 1 : Loading the train data set using pulldata.
    train_data = pulldata.loaddataset('train')
    #print(train_data.columns)


    ## Step 2 : Cleaning up the data as needed.
    try:
        train_data = clean_data(train_data)
    except ValueError:
        print("Oops!  Failure in clean_data method...")
    

    ## Step 3 : Impute the missing elements in the dataset.
    try:
        train_data = impute_data(train_data, 'train')
    except ValueError:
        print("Oops!  Failure in impute_data method...")
    
    ## Step 4 : Prepartion for Model.
    # Split into dependent and independet variables
    y = train_data['Survived'].values
    X = train_data.drop('Survived', axis=1).values
    
    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=38)

    # We will be using the Random Forrest Classifier as asked in question.
    model = RandomForestClassifier(random_state=42) 
    model.fit(X_train, y_train)

    # Predict for test data sample
    model_prediction_train = model.predict(X_test)

    # Confusion_Matrix and Classification Report.
    score = model.score(X_test, y_test)
    print_model_output(y_test, model_prediction_train, score=score)

    # Creation of pkl file for score_model input.
    try: 
        pickle.dump(model, open('model.pkl', 'wb'))
    except:
        raise
    
## As far as train model is concerned, things that could be improved include. Better
## imputation methodologies can be used rather than simple one used in the program.
## Hyperparameter can be used in picking on parameter for the model.