#import pandas as pd 
import numpy as np
import copy
from tqdm import tqdm
from xgboost import XGBClassifier
from algorithm.pseudo_labeling import Pseudo_Labeling
from algorithm.flexmatch import FlexMatch
from algorithm.ups import UPS
from algorithm.csa import CSA
import os
from utilities.utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled,append_acc_early_termination
import pickle
import warnings
warnings.filterwarnings('ignore')

path ='./vector_data/'




FromIndex=0
ToIndex=20
nRepeat=ToIndex-FromIndex



NumIter=5
upper_threshold=0.8
lower_threshold=0.2

# good: 5,6
# need more iter: 2,3,5,7,15,17
for ii, data in enumerate(all_data):
 
    if ii in [0,1,9,11,12]:
        continue
    
    if ii not in [6]:
        continue
 
   
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    #shapes.append(data.shape[0])

    
    
    AccPL=[0]*nRepeat
    AccFlex=[0]*nRepeat
    AccCSA_NoConfidence=[0]*nRepeat
    AccCSA_TotalEnt=[0]*nRepeat
    AccCSA_TotalVar=[0]*nRepeat
    AccCSA_TTest=[0]*nRepeat
    AccUPS=[0]*nRepeat
    
    for tt in range(nRepeat):
        
        np.random.seed(tt)
        # x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled_data(all_data, \
        #                                    _datasetName,dataset_index=ii,random_state=tt)
            
        x_train,y_train, x_test, y_test, x_unlabeled=get_low_perf_train_test_unlabeled(all_data, \
                                            _datasetName,dataset_index=ii,random_state=tt)
        
        # scale the index to 0
        tt=tt-FromIndex

        # print(_datasetName[ii])
        # print(len(np.unique(y_test)))
        # print("feat dim",x_train.shape[1])
        # print(x_train.shape)
        # print(x_unlabeled.shape)
        # print(x_test.shape)
        
        # continue
      
        # method 1    
        pseudo_labeller = Pseudo_Labeling(x_unlabeled,x_test,y_test, # for evaluation purpose
                upper_threshold,lower_threshold,
                num_iters=NumIter,verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        AccPL[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)
        



        pseudo_labeller = FlexMatch(x_unlabeled,x_test,
                y_test,upper_threshold, # for evaluation purpose
                num_iters=NumIter,verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        AccFlex[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

      
      
        #method 4   
        pseudo_labeller = CSA(x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='ttest',
                verbose = True
            )
        pseudo_labeller.fit(x_train, y_train)
      
        AccCSA_TTest[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)


    
        pseudo_labeller = CSA(x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='variance',
                verbose = True
            )
        pseudo_labeller.fit(x_train, y_train)
       
        AccCSA_TotalVar[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        
    
        pseudo_labeller = CSA(x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='no_confidence',
                verbose = True
            )
        pseudo_labeller.fit(x_train, y_train)
        AccCSA_NoConfidence[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        
        pseudo_labeller = CSA(x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='entropy',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        AccCSA_TotalEnt[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        pseudo_labeller = UPS(x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)

        if tt==5:
            print("debug")
       

        AccUPS[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        print(AccUPS[tt])

      
        
    # save result to file
    
    save_folder="results4"
    strFile=save_folder+"/flex_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccFlex)>0:
        np.save(strFile, AccFlex)
    
    strFile=save_folder+"/pl_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccPL)>0:
        np.save(strFile, AccPL)

    strFile=save_folder+"/CSA_TTest_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccCSA_TTest)>0:
        np.save(strFile, AccCSA_TTest)
        
    strFile=save_folder+"/CSA_TotalVar_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccCSA_TotalVar)>0:
        np.save(strFile, AccCSA_TotalVar)
        
    strFile=save_folder+"/CSA_TotalEnt_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccCSA_TotalEnt)>0:
        np.save(strFile, AccCSA_TotalEnt)
        
    strFile=save_folder+"/CSA_NoConfidence_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccCSA_NoConfidence)>0:
        np.save(strFile, AccCSA_NoConfidence)
        
    strFile=save_folder+"/UPS_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccUPS)>0:
        np.save(strFile, AccUPS)
    
    
    continue
