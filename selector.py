# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""


import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import BAT as bat
import WOA as woa
import FFA as ffa
import SSA as ssa

import csv
import numpy
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.metrics import classification_report # for generating a classification report of model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def selector(algo,func_details,popSize,Iter,completeData):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
   
    DatasetSplitRatio=0.25
    DataFile="datasets/"+completeData
    df = pd.read_csv(DataFile, header=None)
    numRowsData=df.shape[0]
    features=df.iloc[0:numRowsData,:-1]
#     print(features.shape)
    features =  features.fillna(features.mean())
    labels=df.iloc[0:numRowsData,-1] 
    labels =  labels.fillna(1)
#     print(labels.shape)
#     print(labels)
#     print('labels.isnull().sum()', labels.isnull().sum())
#     print('features.isnull().sum()', features.isnull().sum())

    le = LabelEncoder()
#     print(type(labels[:1][0]))

    labels= le.fit_transform(labels.astype(str))
#     print("Finished")
###################### KDD Data ##########################
#     DataFile="datasets/"+completeData
#     bin_data = pd.read_csv(DataFile)
#     numRowsData=bin_data.shape[0]
    
#     data = bin_data.iloc[:,0:93] # dataset excluding target attribute (encoded, one-hot-encoded,original)
#     Target = bin_data['intrusion'] # target attribute
    trainInput, testInput, trainOutput, testOutput = train_test_split(features, labels, test_size=DatasetSplitRatio, random_state=42) 

############################################################    
    dataInput  = trainInput
    dataTarget    = trainOutput
    dim=int (features.shape[1])
    if(algo==0):
        x=pso.PSO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==1):
        x=mvo.MVO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==2):
        x=gwo.GWO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==3):
        x=mfo.MFO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==4):
        x=woa.WOA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==5):
        x=ffa.FFA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==6):
        x=bat.BAT(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
    if(algo==7):
        x=ssa.SSA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,dataInput,dataTarget)
 
    # Evaluate MLP classification model based on the training set
#    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
 #   x.trainAcc=trainClassification_results[0]
  #  x.trainTP=trainClassification_results[1]
   # x.trainFN=trainClassification_results[2]
    #x.trainFP=trainClassification_results[3]
    #x.trainTN=trainClassification_results[4]
   
    # Evaluate MLP classification model based on the testing set   
    #testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)
            
    reducedfeatures=[]
    for index in range(0,dim):
        if (x.bestIndividual[index]==1):
            reducedfeatures.append(index)
    reduced_data_train_global=trainInput.iloc[:,reducedfeatures]
    reduced_data_test_global=testInput.iloc[:,reducedfeatures]        
#     print("reducedfeatures")
#     print(reducedfeatures)
#     knn = RandomForestClassifier(random_state=20181224)
    knn = DecisionTreeClassifier(random_state=20181224)
#     knn = SVC(kernel='rbf',verbose=0, random_state=1024)
    #knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(reduced_data_train_global,trainOutput)

    target_pred_train = knn.predict(reduced_data_train_global)
    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc=acc_train
    
    target_pred_test = knn.predict(reduced_data_test_global)
    fpr, tpr, t = roc_curve(testOutput, target_pred_test)
    conf_matrix = confusion_matrix(testOutput, target_pred_test)
 
    
    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc=acc_test


    Sen = float(conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]))
    x.sen=Sen

    Spec = float(conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1]))
    x.spec=Spec
    
    Auc = float(auc(fpr, tpr))
    x.auc=Auc
    
        #print('Test set accuracy: %.2f %%' % (acc * 100))

    #x.testTP=testClassification_results[1]
    #x.testFN=testClassification_results[2]
    #x.testFP=testClassification_results[3]
    #x.testTN=testClassification_results[4] 
    
    
    return x
    
#####################################################################    


