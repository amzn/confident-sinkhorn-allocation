#import pandas as pd 
import numpy as np
import copy
from tqdm import tqdm
from xgboost import XGBClassifier
from pseudo_labelling_algorithms import Pseudo_Labeling,FlexMatch
from pseudo_labelling_algorithms import UPS,CSA
import os
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled,append_acc_early_termination
import pickle
import warnings
warnings.filterwarnings('ignore')

path ='./vector_data/'


# Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/




param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['verbosity'] = 0
param['silent'] = 1
param['seed'] = 0

# create XGBoost instance with default hyper-parameters
xgb = XGBClassifier(**param,use_label_encoder=False)


# load the data
with open('all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)


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
    AccSinkhorn=[0]*nRepeat
    AccCSA_NoConfidence=[0]*nRepeat
    AccCSA_TotalEnt=[0]*nRepeat
    AccCSA_TotalVar=[0]*nRepeat
    AccCSA_TTest=[0]*nRepeat

    AccSinkhornOrig=[0]*nRepeat
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
        # pseudo_labeller = Pseudo_Labeling(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, # for evaluation purpose
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccPL[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)
        



        # pseudo_labeller = FlexMatch(copy.copy(xgb),
        #         x_unlabeled,x_test,
        #         y_test,upper_threshold, # for evaluation purpose
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccFlex[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

      
      
        #method 4   
        # pseudo_labeller = CSA(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='ttest',
        #         verbose = True
        #     )
        # pseudo_labeller.fit(x_train, y_train)
      
        # AccCSA_TTest[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)


        # print(pseudo_labeller.elapse_xgb)
        # print(pseudo_labeller.elapse_ttest)
        # print(pseudo_labeller.elapse_sinkhorn)
        
        
    
        # pseudo_labeller = CSA(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='variance',
        #         verbose = True
        #     )
        # pseudo_labeller.fit(x_train, y_train)
       
        # AccCSA_TotalVar[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        
    
        # pseudo_labeller = CSA(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='no_confidence',
        #         verbose = True
        #     )
        # pseudo_labeller.fit(x_train, y_train)
      
        # AccCSA_NoConfidence[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        
        # pseudo_labeller = CSA(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='entropy',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
    
        # AccCSA_TotalEnt[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        pseudo_labeller = UPS(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)

        if tt==5:
            print("debug")
       

        AccUPS[tt]=append_acc_early_termination(pseudo_labeller.test_acc,NumIter)

        print(AccUPS[tt])

      
      
        #method 7
        # pseudo_labeller = sinkhorn_original(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         flagRepetition=False,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccSinkhornOrig[tt]=pseudo_labeller.test_acc
        # if len(AccSinkhornOrig[tt])<=NumIter:
        #     AccSinkhornOrig[tt] = AccSinkhornOrig[tt] + [AccSinkhornOrig[tt][-1]]*(1+NumIter-len(AccSinkhornOrig[tt]))
        
        
        
    # save result to file
    
    save_folder="results4"
    strFile=save_folder+"/flex_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccFlex)>0:
        np.save(strFile, AccFlex)
    
    strFile=save_folder+"/pl_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccPL)>0:
        np.save(strFile, AccPL)

    strFile=save_folder+"/Sinkhorn_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccSinkhorn)>0:
        np.save(strFile, AccSinkhorn)
        
    
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
