# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
#____________________________________________________________________________________       
def FN1(I,trainInput,trainOutput,dim):
         data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput, trainOutput, test_size=0.25, random_state=42)
         reducedfeatures=[]
         
         for index in range(0,dim):
                if (I[index]==1):
                    reducedfeatures.append(index)
                    
         reduced_data_train_internal=data_train_internal.iloc[:,reducedfeatures]
         reduced_data_test_internal=data_test_internal.iloc[:,reducedfeatures]
#          knn =RandomForestClassifier(random_state=20181224)
         knn = DecisionTreeClassifier(random_state=20181224)
#          knn = SVC(kernel='rbf',verbose=0, random_state=1024)         
         #knn = KNeighborsClassifier(n_neighbors=5)
         knn.fit(reduced_data_train_internal, target_train_internal)
         target_pred_internal = knn.predict(reduced_data_test_internal)
         acc_train = float(accuracy_score(target_test_internal, target_pred_internal))
       
         fitness=0.99*(1-acc_train)+0.01*sum(I)/(dim)

         return fitness
#_____________________________________________________________________       
def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0:["FN1",0,1]

            }
    return param.get(a, "nothing")




