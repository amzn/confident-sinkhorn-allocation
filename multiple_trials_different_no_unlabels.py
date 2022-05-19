import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn import preprocessing
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl#,entropy_pl,prediction_entropy
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import lcb_ucb,sinkhorn_original,sla_no_uncertainty
from pseudo_labelling_algorithms import UPS,csa
from sklearn.preprocessing import StandardScaler
from load_data import load_encode_data
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled,get_train_test_given_different_labelled
import pickle

import warnings
warnings.filterwarnings('ignore')

path ='./vector_data/'


# Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/

def str2num(s, encoder):
    return encoder[s]


param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['verbosity'] = 0
param['silent'] = 1
param['seed'] = 0

#     param["eval_metric"] = "error"
#     param['eta'] = 0.5
#     param['gamma'] = 0.2
#     param['max_depth'] = 3
#     param['min_child_weight']=1
#     param['max_delta_step'] = 0
#     param['subsample']= 0.5
#     param['colsample_bytree']=1
#     
#     param['seed'] = 0
#     param['base_score'] = 0.5


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

no_of_label=600
no_of_unlabel_list=[100,300,500,700,900,1100]
for no_of_unlabel in no_of_unlabel_list:
 
    # data index
    ii=18
    
    data=all_data[ii]

    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    
    AccPL=[0]*nRepeat
    AccFlex=[0]*nRepeat
    AccSinkhorn=[0]*nRepeat
    AccCSA=[0]*nRepeat
    SLA_noconfidence=[0]*nRepeat
    CSA_TotalVar=[0]*nRepeat
    CSA_TTest=[0]*nRepeat
    AccLCBUCB=[0]*nRepeat
    AccUPS=[0]*nRepeat

    for tt in range(nRepeat):
        
        np.random.seed(tt)
        
        x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_given_different_labelled(all_data, \
                _datasetName,dataset_index=ii,random_state=tt,\
                no_of_label=no_of_label,no_of_unlabel=no_of_unlabel)
        
        # scale the index to 0
        tt=tt-FromIndex

        #method 4   
     
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='ttest',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TTest[tt]=pseudo_labeller.test_acc
        if len(CSA_TTest[tt])<=NumIter:
            CSA_TTest[tt] = CSA_TTest[tt] + [CSA_TTest[tt][-1]]*(1+NumIter-len(CSA_TTest[tt]))
        
        
        
    # save result to file
    
    save_folder="results2"
  
    strFile=save_folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}".format(ii,
                                                _datasetName[ii],no_of_label,no_of_unlabel)
    if np.sum(CSA_TTest)>0:
        np.save(strFile, CSA_TTest)
        
     
  