#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CUNY DATA 622 Home Work 2

Created on Mon Oct  8 23:18:11 2018

@author: dilipganesan


When this is called using python score_model.py in the command line, 
this will ingest the .pkl random forest file and apply the model to the locally 
saved scoring dataset csv. There must be data check steps and clear commenting 
for each step inside the .py file. The output for running this file is a 
csv file with the predicted score, as well as a png or text file output that 
contains the model accuracy report (e.g. sklearn's classification report or 
any other way of model evaluation).

"""


# Import functions and packages.

import pull_data as pulldata
import train_model as trainmodel
import pickle
import pandas as pd

def scoring_model():
    # Step 1: Loading the test_data
    test_data = pulldata.loaddataset('test')
    # Assigning the test_data to another variable, bcs we need passenger_id for our
    # final prediction.
    final_pred_data = pd.DataFrame(test_data)
    
    
    # Step 2 : Cleaning the test_data
    try:
        test_data = trainmodel.clean_data(test_data)
    except ValueError:
        print("Oops!  Failure in clean_data method...")
    
    
    # Step 3 : Imputing the test_data
    # Only Fare and Age has missing values in test_data.
    try:
        test_data = trainmodel.impute_data(test_data, 'test')
    except ValueError:
        print("Oops!  Failure in impute_data method...")
    
    
    # Step 4 : Prediction of test_data
    # Load model from the pickle file
    try:
        model = pickle.load(open('model.pkl', 'rb'))
    except:
        raise
    
    # Run prediction
    pred_df = model.predict(test_data)
    
    # Concatenate passenger ID and prediction
    pred_df = pd.concat([final_pred_data['PassengerId'],pd.DataFrame(pred_df)], axis=1)
    pred_df.columns = ['PassengerId', 'Survived']
    
    # Saving the CSV for Kaggle Website.
    try:
        pred_df.to_csv('kaggle_titanic.csv', index=False)
    except:
        raise
    




