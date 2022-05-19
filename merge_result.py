
import numpy as np

#from sklearn.datasets import load_iris,load_breast_cancer,load_digits
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn import preprocessing
from pseudo_labelling_algorithms_multilabel import pseudo_labeling_iterative,flex_pl#,entropy_pl,prediction_entropy
from pseudo_labelling_algorithms_multilabel import lcb_ucb,UPS,sla
#from sklearn.preprocessing import StandardScaler
#from load_data import load_encode_data
import os
#from load_multi_label_data import load_yeast_multilabel,load_emotions_multilabel,load_genbase_multilabel
#from load_multi_label_data import load_corel5k_multilabel
from sklearn.multioutput import MultiOutputClassifier
import pickle
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
#xgb = XGBClassifier(objective='binary:logistic',verbosity = 0)
xgb = XGBClassifier(**param,use_label_encoder=False)



# load the data
with open('all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)



#algorithm_name=['flex','pl','UPS','LCBUCB','CSA_TotalVar','SLA_noconfidence','CSA_TTest']
algorithm_name=['flex','pl','UPS','CSA_TotalVar','SLA_noconfidence','CSA_TTest']
algorithm_name=['CSA_TotalEnt']
#algorithm_name=['flex','pl','Sinkhorn']
#algorithm_name=['flex','pl','Sinkhorn','Sinkhorn_ori']
#algorithm_name=['Sinkhorn']

for ii, data in enumerate(all_data):
  
     
    if ii in [0,1,9,11,12]:
        continue
  
    # if ii not in [14,13,11,8,2,3]:
    #     continue
    
    if ii !=18:
        continue
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])

   
        

    save_folder="results3"
    save_folder2="results3"

    for algo in algorithm_name:
            
        # merging result
        From1=0
        To1=20
        strFile=save_folder+"/{:s}_{:d}_{:s}_{:d}_{:d}.npy".format(algo,ii,_datasetName[ii],From1,To1)
        data1=np.load(strFile,allow_pickle=True)
        #data1b=np.load(strFile,allow_pickle=True)[1]

        From2=20
        To2=40
        strFile=save_folder2+"/{:s}_{:d}_{:s}_{:d}_{:d}.npy".format(algo,ii,_datasetName[ii],From2,To2)
        data2=np.load(strFile,allow_pickle=True)
        #data2b=np.load(strFile,allow_pickle=True)[1]

        strFile=save_folder+"/{:s}_{:d}_{:s}".format(algo,ii,_datasetName[ii])
        data=np.vstack((data1,data2))
        #datab=np.vstack((data1b,data2b))

        np.save(strFile, data)
    

    
   