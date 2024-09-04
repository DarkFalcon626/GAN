# -*- coding: utf-8 -*-
"""
This file reduces the size of a dataset and
creates a new file containing the reduced
dataset. This dataset is randomized each time.
-----------------------------------------------
Created on Wed Sep  4 01:34:45 2024
-----------------------------------------------
@author: Andrew Francey
"""

import pickle
import os
import random as rand

## The datapath of the dataset to reduce.
datapath = os.path.dirname(__file__) + '\\Dataset\\64images.pickle'

## Loads in the pickled dataset.
with open(datapath, 'rb') as datafile:
    data = pickle.load(datafile)
datafile.close()

## Precentage to reduce the dataset by.
reduction = int(input("Percentage to reduce dataset by: "))/100

## Size of the dataset.
data_size = len(data)

## Size of the reduced dataset
new_size = int(data_size*(1-reduction))
print(new_size)

## Initialize the new data list and a list to store the used indices.
new_data = []
used_index = []

## Pick random unused data points from the original dataset.
while len(new_data) < new_size:
    
    i = int(rand.uniform(0, data_size))
    
    if not(i in used_index):
        new_data.append(data[i])
        used_index.append(i)

## Determine what to name this new dataset.
i = 0
DatasetExists = True
while DatasetExists:
    DatasetPath = datapath[:-7] + 'Reduced' + str(i) + datapath[-7:]
    
    DatasetExists = os.path.exists(DatasetPath)
    
    i += 1

## Save the new dataset.
with open(DatasetPath, 'wb') as newfile:
    pickle.dump(new_data, newfile)

newfile.close()