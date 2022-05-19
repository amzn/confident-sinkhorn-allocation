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
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl#,entropy_pl,prediction_entropy
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import lcb_ucb,sinkhorn_original,sla_no_uncertainty
from pseudo_labelling_algorithms import UPS,csa
from sklearn.preprocessing import StandardScaler
from load_data import load_encode_data
import os
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled

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
#xgb = XGBClassifier(objective='binary:logistic',verbosity = 0)
xgb = XGBClassifier(**param,use_label_encoder=False)

log_reg = SGDClassifier(loss = 'log', n_jobs = -1, alpha = 1e-5)





all_data = []
s_dataname = []
shapes = []

_datasetName =["cjs","hill-valley","segment_2310_20","wdbc_569_31","steel-plates-fault",
          "analcatdata_authorship","synthetic_control_6c","vehicle_846_19","German-credit",
          "gina_agnostic_no","madelon_no","texture","gas_drift","dna_no"
           ] #
for ii in range(len(_datasetName)):
    temp = pd.read_csv(os.path.join(path ,_datasetName[ii]+".csv"))
    if temp.shape[0] <= 30000000:
        print(_datasetName[ii])
        all_data.append(temp)
        s_dataname.append(_datasetName[ii]+".csv")
        
        
# load data from UCI
path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("car")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

    
path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("kr_vs_kp")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("agaricus-lepiota")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))




# data = load_iris()
# X=data['data']
# Y = data['target']
# _datasetName.append("iris")
# all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

data=load_breast_cancer() # binary
X=data['data']
Y = data['target']
_datasetName.append("breast_cancer")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))


data=load_digits()
X=data['data']
Y = data['target']
_datasetName.append("digits")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))


FromIndex=0
ToIndex=20
nRepeat=ToIndex-FromIndex



upper_threshold=0.8
lower_threshold=0.2

# good: 5,6
# need more iter: 2,3,5,7,15,17

NumIter_List=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for NumIter in NumIter_List:
 
    if ii in [0,1,9,11,12,16]:
        continue
    
    # data index
    ii=5
    
    data=all_data[ii]
    
    # if ii not in [1,2,4,12,13,15]:
    #     continue
 
    # if ii not in [1,4,5,6,7,8,15]:
    #     continue
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    shapes.append(data.shape[0])

    if ii<14:
        _dic = list(set(data.values[:, -1]))
        num_labels = len(_dic)
        encoder = {}
        for i in range(len(_dic)):
            encoder[_dic[i]] = i
    
        # shuffle original dataset
        data = data.sample(frac=1,random_state=42)
        X = data.values[:, :-1]
        # X = scale(X)  # scale the X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = np.array([str2num(s, encoder) for s in data.values[:, -1]])
    else:
        X = data[:, :-1]
        Y=data[:,-1]
        
        
    
    AccPL=[0]*nRepeat
    AccFlex=[0]*nRepeat
    AccSinkhorn=[0]*nRepeat
    AccCSA=[0]*nRepeat
    SLA_noconfidence=[0]*nRepeat
    CSA_TotalVar=[0]*nRepeat
    CSA_TTest=[0]*nRepeat
    CESLATeacher=[0]*nRepeat
    AccSinkhornOrig=[0]*nRepeat
    AccSinkhornT=[0]*nRepeat
    AccLCBUCB=[0]*nRepeat
    AccUPS=[0]*nRepeat
    AccUncEntPred=[0]*nRepeat
    
    for tt in range(nRepeat):
        
        np.random.seed(tt)
        

        x_train,y_train, x_test, y_test, x_unlabeled=get_low_perf_train_test_unlabeled(all_data, \
                                            _datasetName,dataset_index=ii,random_state=tt)
        
        # scale the index to 0
        tt=tt-FromIndex



      
        #method 4   
        # pseudo_labeller = csa_ttest(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccSinkhorn[tt]=pseudo_labeller.test_acc
        # if len(AccSinkhorn[tt])<=NumIter:
        #     AccSinkhorn[tt] = AccSinkhorn[tt] + [AccSinkhorn[tt][-1]]*(1+NumIter-len(AccSinkhorn[tt]))
            
            
            
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccCSA[tt]=pseudo_labeller.test_acc
        # if len(AccCSA[tt])<=NumIter:
        #     AccCSA[tt] = AccCSA[tt] + [AccCSA[tt][-1]]*(1+NumIter-len(AccCSA[tt]))
            
   
       
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
        
        
        
        # pseudo_labeller = csa_totalvar(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TotalVar[tt]=pseudo_labeller.test_acc
        # if len(CSA_TotalVar[tt])<=NumIter:
        #     CSA_TotalVar[tt] = CSA_TotalVar[tt] + [CSA_TotalVar[tt][-1]]*(1+NumIter-len(CSA_TotalVar[tt]))
            
    
    # save result to file
    
  
    save_folder="results2"
  
    strFile=save_folder+"/CSA_TTest_{:d}_{:s}_{:d}_{:d}_Iter_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex,NumIter)
    if np.sum(CSA_TTest)>0:
        np.save(strFile, CSA_TTest)
        
     
  