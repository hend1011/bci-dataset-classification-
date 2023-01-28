# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:32:58 2021

@author: Mohanmmed
"""

import warnings
warnings.filterwarnings("ignore")
import csv
import numpy
import time
import selector as slctr
import fitnessFUNs

# Select optimizers

PSO= False 
MVO= False
GWO = False
MFO= False
WOA= False
FFA=False
BAT=False
SSA=False
SSAGWO=True

ISSA = False




optimizer=[PSO,MVO,GWO,MFO, WOA,FFA,BAT,SSA,SSAGWO]
# optimizer=[GWO,WOA,SSA]
datasets=["Zoo"]
#benchmarkfunc=[Fs1,Fs2,Fs3,Fs4,Fs5,Fs6,Fs7,Fs8,Fs9,Fs10] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=20

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 10
Iterations= 50

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated file name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader1=[]
CnvgHeader2=[]


for l in range(0,Iterations):
	CnvgHeader1.append("Iter"+str(l+1))

for l in range(0,Iterations):
	CnvgHeader2.append("Iter"+str(l+1))


for j in range (0, len(datasets)):        # specfiy the number of the datasets
    for i in range (0, len(optimizer)):
    
        if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                
                #func_details=["costNN",-1,1]
                func_details=fitnessFUNs.getFunctionDetails(0)
                completeData=datasets[j]+".csv"
                x=slctr.selector(i,func_details,PopulationSize,Iterations,completeData)
                  
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime","trainAcc","testAcc",,"Sensitivity","Specificity","AUC"],CnvgHeader1,CnvgHeader1])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.optimizer,datasets[j],x.objfname,k+1,x.startTime,x.endTime,x.executionTime,x.trainAcc,x.testAcc,x.sen,x.spec,x.auc],x.convergence1,x.convergence2])
                        writer.writerow(a)
                    out.close()
                Flag=True # at least one experiment
                
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        