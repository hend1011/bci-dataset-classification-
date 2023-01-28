#[2020]-"Time-varying hierarchical chains of salps with random weight networks for feature selection"

import numpy as np
from numpy.random import rand
import fitnessFUNs
import time
from solution import solution
import time
import transfer_functions_benchmark

# def init_position(lb, ub, N, dim):
#     X = np.zeros([N, dim], dtype='float')
#     for i in range(N):
#         for d in range(dim):
#             X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
#     return X
def init_position(lb, ub, N, dim):
#     X = np.zeros([N, dim], dtype='float')
    X=np.random.randint(2, size=(N,dim))
#     for i in range(N):
#         for d in range(dim):
#             X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            temp = transfer_functions_benchmark.s2(X[i,d])
            if temp<np.random.uniform(0,1):
                Xbin[i,d] = 0
            else:
                Xbin[i,d] = 1
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x


#--- transfer function (10)
def transfer_function(x):
    Tx = 1 / (1 + np.exp(-x))
    
    return Tx


def SSA(objf,lb,ub,dim,SearchAgents_no,max_iter,trainInput,trainOutput):
    # Parameters
    ub             = ub
    lb             = lb
    thres          = 0.5
    Mp    = 0.5    # mutation probability    
    N              = SearchAgents_no
    xtrain    = trainInput
    ytrain    = trainOutput
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    #--- Binary conversion
    X     = binary_conversion(X, thres, N, dim)    
    n_min = int(SearchAgents_no/5)
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='int')
    fitF  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    curve_f = np.zeros([1, max_iter], dtype='int')
    t     = 0
    
    s=solution()
    print('SSA New is optimizing  "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    while t < max_iter:

        
#         print("Iteration:", t + 1)
#         print("Best (TVBSSA):", curve[0,t])
        
 	    # Compute coefficient, c1 (3)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        
        #--- update number of leaders (12)
        
        for i in range(N):          
            #--- leader update
            if i < N / 2:  
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand() 
                    c3 = rand()
              	    # Leader update (2)
                    if c3 >= 0.5: 
                        Xn = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                    else:
                        Xn = Xf[0,d] - c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                
                    #--- transfer function
                    Tx = transfer_functions_benchmark.s2(Xn)
                    
                    #--- binary update (11)
                    if Tx < np.random.uniform(0,1):
                        X[i,d] = 0
                    else:
                        X[i,d] = 1
                
            #--- Salp update
            elif i >= N / 2 and i < N + 1:
                for d in range(dim):
                    # Salp update by following front salp (4)
                    Xn = (X[i,d] + X[i-1, d]) / 2
                    # Boundary
                    Xn = boundary(Xn, lb[0,d], ub[0,d]) 
                    #--- Binary conversion
                    temp = transfer_functions_benchmark.s2(Xn)
                    if temp < np.random.uniform(0,1):
                        X[i,d] = 0
                    else:
                        X[i,d] = 1
        
        # Fitness
        for i in range(N):
            fit[i,0] = objf(X[i,:],xtrain,ytrain,dim)
            
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
        
        # Store result
        curve[0,t] = fitF.copy()
        
        Gbin       = binary_conversion(Xf, thres, 1, dim) 
        Gbin       = Gbin.reshape(dim)
        pos        = np.asarray(range(0, dim))    
        sel_index  = pos[Gbin == 1]
        num_feat   = len(sel_index)
        curve_f[0,t] = num_feat 
        
        t += 1

        
#         # Linear Population Size Reduction
#         N = round(SearchAgents_no + t * ((n_min - SearchAgents_no)/max_iter))
        
        
#         #--- two phase mutation: first phase
#         # find index of 1
#         idx        = np.where(Xf == 1)
#         idx1       = idx[1]
#         Xmut1      = np.zeros([1, dim], dtype='int')
#         Xmut1[0,:] = Xf[0,:]
#         for d in range(len(idx1)):
#             r = rand()
#             if r < Mp:
#                 Xmut1[0, idx1[d]] = 0
#                 while np.sum(Xmut1[0,:])==0:
#                     Xmut1[0,:]=np.random.randint(2, size=(1,dim))
#                 Fnew1 = objf(Xmut1[0,:],xtrain,ytrain,dim)
#                 #Fun(xtrain, ytrain, Xmut1[0,:], opts)
#                 if Fnew1 < fitF:
#                     Xf[0,:] = Xmut1[0,:]
#                     fitF    = Fnew1
                    
#         #--- two phase mutation: second phase        
#         # find index of 0
#         idx        = np.where(Xf == 0)
#         idx0       = idx[1]
#         Xmut2      = np.zeros([1, dim], dtype='int')
#         Xmut2[0,:] = Xf[0,:]    
#         for d in range(len(idx0)):
#             r = rand()
#             if r < Mp:
#                 Xmut2[0, idx0[d]] = 1
#                 while np.sum(Xmut2[0,:])==0:
#                     Xmut2[0,:]=np.random.randint(2, size=(1,dim)) 
#                 Fnew2 = objf(Xmut2[0,:],xtrain,ytrain,dim)
#                 #Fun(xtrain, ytrain, Xmut2[0,:], opts)
#                 if Fnew2 < fitF:
#                     Xf[0,:] = Xmut2[0,:]
#                     fitF    = Fnew2                    
                
    # Best feature subset
    Gbin       = Xf[0,:] 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tvbssa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    
    timerEnd=time.time() 
    
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=sel_index
    s.convergence1=curve.reshape(max_iter)
    s.convergence2=curve_f.reshape(max_iter)

    s.optimizer="SSA"
    s.objfname=objf.__name__
    return s  

