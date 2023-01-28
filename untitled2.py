# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 11:57:16 2021

@author: Mohanmmed
"""
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import pandas as pd
import numpy as np

DataFile="datasets/vehicle.csv"
df = pd.read_csv(DataFile, header=None)
numRowsData=df.shape[0]
features=df.iloc[0:numRowsData,:-1]
labels=df.iloc[0:numRowsData,-1]  
le = LabelEncoder()
labels= le.fit_transform(labels)
