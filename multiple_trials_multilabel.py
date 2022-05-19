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
from pseudo_labelling_algorithms_multilabel import pseudo_labeling_iterative,flex_pl,csa#,entropy_pl,prediction_entropy
from pseudo_labelling_algorithms_multilabel import lcb_ucb,UPS,sla,sla_noconfidence
from sklearn.preprocessing import StandardScaler
from load_data import load_encode_data
import os
from load_multi_label_data import load_yeast_multilabel,load_emotions_multilabel,load_genbase_multilabel
from load_multi_label_data import load_corel5k_multilabel
from sklearn.multioutput import MultiOutputClassifier

import warnings
warnings.filterwarnings('ignore')


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

data=load_yeast_multilabel(path) # binary
X=data['data']
Y = data['target']
_datasetName.append("breast_cancer")
all_data.append( data )



data=load_emotions_multilabel(path) # binary
X=data['data']
Y = data['target']
_datasetName.append("emotions")
all_data.append( data )


data=load_genbase_multilabel(path) # binary
X=data['data']
Y = data['target']
_datasetName.append("genbase")
all_data.append( data )


# data=load_corel5k_multilabel() # binary
# X=data['data']
# Y = data['target']
# _datasetName.append("corel5k")
# all_data.append( data )


FromIndex=0
ToIndex=15
nRepeat=ToIndex-FromIndex


    
#nRepeat=15
NumIter=5
upper_threshold=0.8
lower_threshold=0.15



for ii, data in enumerate(all_data):
  
    if ii in [2]:
        fraction_allocation=1
    elif ii in [1]:
        fraction_allocation=1
    else:
        fraction_allocation=0.5
        
    if ii!=2:
        continue
    
  
  
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])

    X = data['data']
    Y=data['target']
        

    
    AccPL=[0]*nRepeat
    AccFlex=[0]*nRepeat
    AccSinkhorn=[0]*nRepeat
    AccSinkhorn_one=[0]*nRepeat
    CSA_TotalVar=[0]*nRepeat
    CSA_TotalVar2=[0]*nRepeat
    CSA_TotalEnt2=[0]*nRepeat
    CSA_TotalEnt=[0]*nRepeat

    SLA_NoConfidence=[0]*nRepeat

    AccSLATeacher=[0]*nRepeat
    CESLATeacher=[0]*nRepeat
    AccLCBUCB=[0]*nRepeat
    AccUPS=[0]*nRepeat
    
    HammingLossPL=[0]*nRepeat
    HammingLossFlex=[0]*nRepeat
    HammingLossSinkhorn=[0]*nRepeat
    HammingLossSinkhorn_one=[0]*nRepeat
    #AccSLATeacher=[0]*nRepeat
    #CESLATeacher=[0]*nRepeat
    HammingLossLCBUCB=[0]*nRepeat
    HammingLossUPS=[0]*nRepeat
    
    
    for tt in range(FromIndex,ToIndex):
        
      
       
        np.random.seed(tt)
        
        # 30% train 70%test
        if ii==1:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=tt)
            
            # 25% train 25% unlabelled
            x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, 
                                                          test_size=0.5, random_state=tt)
        elif ii==2:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=tt)
            
            # 25% train 25% unlabelled
            x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, 
                                                          test_size=0.7, random_state=tt)
        else:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=tt)
            
            # 25% train 25% unlabelled
            x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, 
                                                          test_size=0.7, random_state=tt)
            
        
        p = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[p], y_train[p]
        
        p = np.random.permutation(x_unlabeled.shape[0])
        x_unlabeled, y_unlabeled = x_unlabeled[p], y_unlabeled[p]
        
        
    
        # pseudo_labeling_iterative
        # flex_pl
        # flex_pl_teacher_student
        # entropy_pl
        # pl_teacher_student_stop
        
        tt=tt-FromIndex
        # method 1    
        # pseudo_labeller = pseudo_labeling_iterative(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, # for evaluation purpose
        #         upper_threshold,lower_threshold,fraction_allocation,
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccPL[tt]=pseudo_labeller.test_prec
        # HammingLossPL[tt]=pseudo_labeller.test_hl

        # if len(AccPL[tt])<=NumIter:
        #     AccPL[tt] = AccPL[tt] + [AccPL[tt][-1]]*(1+NumIter-len(AccPL[tt]))
        #     HammingLossPL[tt] = HammingLossPL[tt] + [HammingLossPL[tt][-1]]*(1+NumIter-len(HammingLossPL[tt]))

        
        # method 2    
        # pseudo_labeller = flex_pl(copy.copy(xgb),
        #         x_unlabeled,x_test,
        #         y_test,upper_threshold,lower_threshold,fraction_allocation, # for evaluation purpose
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccFlex[tt]=pseudo_labeller.test_prec
        # HammingLossFlex[tt]=pseudo_labeller.test_hl

        # if len(AccFlex[tt])<=NumIter:
        #     AccFlex[tt] = AccFlex[tt] + [AccFlex[tt][-1]]*(1+NumIter-len(AccFlex[tt]))
        #     HammingLossFlex[tt] = HammingLossFlex[tt] + [HammingLossFlex[tt][-1]]*(1+NumIter-len(HammingLossFlex[tt]))

   
        #method 4   
        # pseudo_labeller = sla(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,fraction_allocation,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccSinkhorn[tt]=pseudo_labeller.test_prec
        # HammingLossSinkhorn[tt]=pseudo_labeller.test_hl
        # if len(AccSinkhorn[tt])<=NumIter:
        #     AccSinkhorn[tt] = AccSinkhorn[tt] + [AccSinkhorn[tt][-1]]*(1+NumIter-len(AccSinkhorn[tt]))
        #     HammingLossSinkhorn[tt] = HammingLossSinkhorn[tt] + [HammingLossSinkhorn[tt][-1]]*(1+NumIter-len(HammingLossSinkhorn[tt]))
        
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,fraction_allocation,
                num_iters=NumIter,
                confidence_choice='variance',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TotalVar2[tt]=pseudo_labeller.test_prec
        if len(CSA_TotalVar2[tt])<=NumIter:
            CSA_TotalVar2[tt] = CSA_TotalVar2[tt] + [CSA_TotalVar2[tt][-1]]*(1+NumIter-len(CSA_TotalVar2[tt]))
        
        
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,fraction_allocation,
                num_iters=NumIter,
                confidence_choice='neg_variance',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TotalVar2[tt]=pseudo_labeller.test_prec
        if len(CSA_TotalVar2[tt])<=NumIter:
            CSA_TotalVar2[tt] = CSA_TotalVar2[tt] + [CSA_TotalVar2[tt][-1]]*(1+NumIter-len(CSA_TotalVar2[tt]))
        
        
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,fraction_allocation,
                num_iters=NumIter,
                confidence_choice='entropy',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TotalEnt[tt]=pseudo_labeller.test_prec
        if len(CSA_TotalEnt[tt])<=NumIter:
            CSA_TotalEnt[tt] = CSA_TotalEnt[tt] + [CSA_TotalEnt[tt][-1]]*(1+NumIter-len(CSA_TotalEnt[tt]))
        
        
        
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,fraction_allocation,
                num_iters=NumIter,
                confidence_choice='neg_entropy',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TotalEnt2[tt]=pseudo_labeller.test_prec
        if len(CSA_TotalEnt2[tt])<=NumIter:
            CSA_TotalEnt2[tt] = CSA_TotalEnt2[tt] + [CSA_TotalEnt2[tt][-1]]*(1+NumIter-len(CSA_TotalEnt2[tt]))
        

          
        # pseudo_labeller = sla_noconfidence(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,fraction_allocation,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # SLA_NoConfidence[tt]=pseudo_labeller.test_prec
        # if len(SLA_NoConfidence[tt])<=NumIter:
        #     SLA_NoConfidence[tt] = SLA_NoConfidence[tt] + [SLA_NoConfidence[tt][-1]]*(1+NumIter-len(SLA_NoConfidence[tt]))
        



        # pseudo_labeller = UPS(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,fraction_allocation,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccUPS[tt]=pseudo_labeller.test_prec
        # if len(AccUPS[tt])<=NumIter:
        #     AccUPS[tt] = AccUPS[tt] + [AccUPS[tt][-1]]*(1+NumIter-len(AccUPS[tt]))
            
  
    
  
    
        # pseudo_labeller = lcb_ucb(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,fraction_allocation,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccLCBUCB[tt]=pseudo_labeller.test_prec
        # if len(AccLCBUCB[tt])<=NumIter:
        #     AccLCBUCB[tt] = AccLCBUCB[tt] + [AccLCBUCB[tt][-1]]*(1+NumIter-len(AccLCBUCB[tt]))
            

    
        
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        # ax2.plot( np.asarray(pseudo_labeller.ce_difference_list)*100,'b-.',label="CE Difference")
        # ax2.set_ylabel("Student-Teacher Cross-Entropy",color='b')
        # ax2.tick_params(axis='y', labelcolor='b')
        
    # save result to file
    
    save_folder="multilabel_results"
    
    strFile=save_folder+"/flex_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccFlex)>0:
        np.save(strFile, [AccFlex,HammingLossFlex])
    
    strFile=save_folder+"/pl_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccPL)>0:
        np.save(strFile, [AccPL,HammingLossPL])
  
    
    strFile=save_folder+"/Sinkhorn_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(AccSinkhorn))>0:
        np.save(strFile, [AccSinkhorn,HammingLossSinkhorn])
        
        
    strFile=save_folder+"/SLA_NoConfidence_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(SLA_NoConfidence))>0:
        np.save(strFile, SLA_NoConfidence)
        
        
    strFile=save_folder+"/CSA_TotalEnt_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(CSA_TotalEnt))>0:
        np.save(strFile, CSA_TotalEnt)
        
        
    strFile=save_folder+"/CSA_TotalEnt2_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(CSA_TotalEnt2))>0:
        np.save(strFile, CSA_TotalEnt2)
        
    strFile=save_folder+"/CSA_TotalVar_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(CSA_TotalVar))>0:
        np.save(strFile, CSA_TotalVar)
        
    strFile=save_folder+"/CSA_TotalVar2_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(np.asarray(CSA_TotalVar2))>0:
        np.save(strFile, CSA_TotalVar2)
    
    strFile=save_folder+"/UPS_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccUPS)>0:
        np.save(strFile, AccUPS)
    

    strFile=save_folder+"/LCBUCB_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccLCBUCB)>0:
        np.save(strFile, AccLCBUCB)
        
# merging result
# From1=0
# To1=15
# strFile="multilabel_results/sinkhorn_one_{:d}_{:s}_{:d}_{:d}.npy".format(ii,_datasetName[ii],From1,To1)
# data1=np.load(strFile,allow_pickle=True)

# From2=15
# To2=30
# strFile="multilabel_results/sinkhorn_one_{:d}_{:s}_{:d}_{:d}.npy".format(ii,_datasetName[ii],From2,To2)
# data2=np.load(strFile,allow_pickle=True)

# strFile=save_folder+"/Sinkhorn_one_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
# data=np.vstack((data1,data2))
# np.save(strFile, data)


    
   