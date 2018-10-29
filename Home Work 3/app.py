# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:26:22 2018

@author: ganesand

This file the main file, which calls the train_model, which loads the titanic data
and builds the model. The next step is scoring the model by executing the scoring_model
method in the score_model class.

"""

import train_model as training
import score_model as testing

print('######Building model########')
training.building_model()

print('#######Scoring model#######')
testing.scoring_model()

print('Execution Succesfull')

