#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:12:53 2018

@author: dilipganesan
"""

## Import the needed functions.
import pandas as pd


## loading the data from kaggle.
def loaddataset(target):
    if target == 'train':
        df = pd.read_csv(train_url)
    elif target == 'test':
        df = pd.read_csv(test_url)
    else:
        raise ValueError('Not the correct target')
        
    return df

train_url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
test_url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv"


## This class is used as a safety net. Please refere new_pull.py. That is the latest and
## greatest file. It meets the requirement as asked in home work.
