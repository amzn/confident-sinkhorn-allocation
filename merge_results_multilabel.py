import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import SGDClassifier 
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn import preprocessing
from pseudo_labelling_algorithms_multilabel import pseudo_labeling_iterative,flex_pl#,entropy_pl,prediction_entropy
from pseudo_labelling_algorithms_multilabel import lcb_ucb,UPS,sla
from sklearn.preprocessing import StandardScaler
from load_data import load_encode_data
import os
from load_multi_label_data import load_yeast_multilabel,load_emotions_multilabel,load_genbase_multilabel
from load_multi_label_data import load_corel5k_multilabel
from sklearn.multioutput import MultiOutputClassifier

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    
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

# create XGBoost instance with default hyper-parameters
#xgb = XGBClassifier(objective='binary:logistic',verbosity = 0)
#xgb = XGBClassifier(**param,use_label_encoder=False)
xgb=MultiOutputClassifier(XGBClassifier(**param,use_label_encoder=False))
all_data = []
s_dataname = []




_datasetName=[]

data=load_yeast_multilabel() # binary
X=data['data']
Y = data['target']
_datasetName.append("breast_cancer")
all_data.append( data )



data=load_emotions_multilabel() # binary
X=data['data']
Y = data['target']
_datasetName.append("emotions")
all_data.append( data )


data=load_genbase_multilabel() # binary
X=data['data']
Y = data['target']
_datasetName.append("genbase")
all_data.append( data )


# data=load_corel5k_multilabel() # binary
# X=data['data']
# Y = data['target']
# _datasetName.append("corel5k")
# all_data.append( data )


algorithm_name=['flex','pl','Sinkhorn','SLA_NoConfidence','UPS','CSA_TotalVar']
#algorithm_name=['flex','pl','Sinkhorn']

for ii, data in enumerate(all_data):
  
        
    if ii!=2:
        continue
    
  
  
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])

   
        

    save_folder="multilabel_results"
    
    for algo in algorithm_name:
        
        data1b=[]
        data2b=[]
        
        # merging result
        From1=0
        To1=15
        strFile=save_folder+"/{:s}_{:d}_{:s}_{:d}_{:d}.npy".format(algo,ii,_datasetName[ii],From1,To1)
        temp=np.load(strFile)
        
        if temp.shape[0]>2:
            data1=temp
        else:
            data1=temp[0]
            data1b=temp[1]

        From2=15
        To2=30
        strFile=save_folder+"/{:s}_{:d}_{:s}_{:d}_{:d}.npy".format(algo,ii,_datasetName[ii],From2,To2)
        temp=np.load(strFile)
        
        if temp.shape[0]>2:
            data2=temp
        else:
            data2=temp[0]
            data2b=temp[1]

        strFile=save_folder+"/{:s}_{:d}_{:s}".format(algo,ii,_datasetName[ii])
        data=np.vstack((data1,data2))
        datab=np.vstack((data1b,data2b))

        if len(data1b)==0 :
            np.save(strFile, data)
        else:
            np.save(strFile, [data, datab])
    

    
   