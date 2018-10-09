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
        df = pd.read_csv('train.csv')
    elif target == 'test':
        df = pd.read_csv('test.csv')
    else:
        raise ValueError('Not the correct target')
        
    return df

## Validation of data set