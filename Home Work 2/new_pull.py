#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:46:56 2018

@author: dilipganesan
"""

import subprocess
import io
import pandas as pd
import os
import json

# Json file with credentials are stored in github with gitignore.
with open("kaggle.json") as json_file:
    json_data = json.load(json_file)
    #print(json_data['username'])
    #print(json_data['key'])

os.environ["KAGGLE_USERNAME"] = json_data['username']
os.environ["KAGGLE_KEY"] = json_data['key']
cmd = ["kaggle", "competitions", "download", "-f", "train.csv","-o", "titanic"]
cmd2 = ["kaggle", "competitions", "download", "-f", "test.csv","-o", "titanic"]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
csv = io.StringIO(process.stdout.read().decode())
data = pd.read_csv(csv)
csv.close()

process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE)
csv2 = io.StringIO(process2.stdout.read().decode())
data2 = pd.read_csv(csv2)
csv2.close()

## loading the data from kaggle.
def loaddataset(target):
    if target == 'train':
        df = pd.read_csv('train.csv')
    elif target == 'test':
        df = pd.read_csv('test.csv')
    else:
        raise ValueError('Not the correct target')
        
    return df


## We could have added some validation to the dataset, making sure the rows and columns
## are as expected. Making sure the Pred column is not there in test.csv
