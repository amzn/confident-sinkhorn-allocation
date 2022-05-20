#import pandas as pd 
import numpy as np
import copy
from tqdm import tqdm
from xgboost import XGBClassifier
from pseudo_labelling_algorithms import pseudo_labeling_iterative,FlexMatch
from pseudo_labelling_algorithms import UPS,csa
import os
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled
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
    
    if ii not in [18]:
        continue
 
   
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    #shapes.append(data.shape[0])

    
    
    AccPL=[0]*nRepeat
    AccFlex=[0]*nRepeat
    AccSinkhorn=[0]*nRepeat
    AccCSA=[0]*nRepeat
    SLA_noconfidence=[0]*nRepeat
    CSA_TotalEnt=[0]*nRepeat
    CSA_TotalVar=[0]*nRepeat
    CSA_TotalVar_Ent=[0]*nRepeat
    CSA_TTest=[0]*nRepeat
    CSA_TotalEnt2=[0]*nRepeat
    CSA_TotalVar2=[0]*nRepeat
    CSA_TTest2=[0]*nRepeat

   
    AccSinkhornOrig=[0]*nRepeat
    AccSinkhornT=[0]*nRepeat
    AccLCBUCB=[0]*nRepeat
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
        # pseudo_labeling_iterative
        # flex_pl
        # flex_pl_teacher_student
        # entropy_pl
        # pl_teacher_student_stop
        
        # method 1    
        # pseudo_labeller = pseudo_labeling_iterative(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, # for evaluation purpose
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccPL[tt]=pseudo_labeller.test_acc
        # if len(AccPL[tt])<=NumIter:
        #     AccPL[tt] = AccPL[tt] + [AccPL[tt][-1]]*(1+NumIter-len(AccPL[tt]))
            
        
        # pseudo_labeller = flex_pl(copy.copy(xgb),
        #         x_unlabeled,x_test,
        #         y_test,upper_threshold, # for evaluation purpose
        #         num_iters=NumIter,verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccFlex[tt]=pseudo_labeller.test_acc
        # if len(AccFlex[tt])<=NumIter:
        #     AccFlex[tt] = AccFlex[tt] + [AccFlex[tt][-1]]*(1+NumIter-len(AccFlex[tt]))
            
      
      
        #method 4   
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='ttest',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TTest[tt]=pseudo_labeller.test_acc
        # if len(CSA_TTest[tt])<=NumIter:
        #     CSA_TTest[tt] = CSA_TTest[tt] + [CSA_TTest[tt][-1]]*(1+NumIter-len(CSA_TTest[tt]))
        
        # print(pseudo_labeller.elapse_xgb)
        # print(pseudo_labeller.elapse_ttest)
        # print(pseudo_labeller.elapse_sinkhorn)
        
        
        
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='neg_ttest',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TTest2[tt]=pseudo_labeller.test_acc
        # if len(CSA_TTest2[tt])<=NumIter:
        #     CSA_TTest2[tt] = CSA_TTest2[tt] + [CSA_TTest2[tt][-1]]*(1+NumIter-len(CSA_TTest2[tt]))
    
    
    
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='variance',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TotalVar[tt]=pseudo_labeller.test_acc
        # if len(CSA_TotalVar[tt])<=NumIter:
        #     CSA_TotalVar[tt] = CSA_TotalVar[tt] + [CSA_TotalVar[tt][-1]]*(1+NumIter-len(CSA_TotalVar[tt]))
        
        
    
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='no',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # SLA_noconfidence[tt]=pseudo_labeller.test_acc
        # if len(SLA_noconfidence[tt])<=NumIter:
        #     SLA_noconfidence[tt] = SLA_noconfidence[tt] + [SLA_noconfidence[tt][-1]]*(1+NumIter-len(SLA_noconfidence[tt]))
        
        
        
        # pseudo_labeller = csa(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         confidence_choice='neg_variance',
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TotalVar2[tt]=pseudo_labeller.test_acc
        # if len(CSA_TotalVar2[tt])<=NumIter:
        #     CSA_TotalVar2[tt] = CSA_TotalVar2[tt] + [CSA_TotalVar2[tt][-1]]*(1+NumIter-len(CSA_TotalVar2[tt]))
            
        
        pseudo_labeller = csa(copy.copy(xgb),
                x_unlabeled,x_test,y_test, 
                upper_threshold,lower_threshold,
                num_iters=NumIter,
                confidence_choice='entropy',
                verbose = True
            )
        
        pseudo_labeller.fit(x_train, y_train)
        CSA_TotalEnt[tt]=pseudo_labeller.test_acc
        if len(CSA_TotalEnt[tt])<=NumIter:
            CSA_TotalEnt[tt] = CSA_TotalEnt[tt] + [CSA_TotalEnt[tt][-1]]*(1+NumIter-len(CSA_TotalEnt[tt]))
            
        # pseudo_labeller = csa_totalentropy2(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TotalEnt2[tt]=pseudo_labeller.test_acc
        # if len(CSA_TotalEnt2[tt])<=NumIter:
        #     CSA_TotalEnt2[tt] = CSA_TotalEnt2[tt] + [CSA_TotalEnt2[tt][-1]]*(1+NumIter-len(CSA_TotalEnt2[tt]))
            
            
             
        # pseudo_labeller = csa_totalvar_ent(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # CSA_TotalVar_Ent[tt]=pseudo_labeller.test_acc
        # if len(CSA_TotalVar_Ent[tt])<=NumIter:
        #     CSA_TotalVar_Ent[tt] = CSA_TotalVar_Ent[tt] + [CSA_TotalVar_Ent[tt][-1]]*(1+NumIter-len(CSA_TotalVar_Ent[tt]))
            
      
            
      
        
        # print(pseudo_labeller.len_unlabels)
        # print(pseudo_labeller.len_accepted_ttest)
        # print(pseudo_labeller.len_selected)
        
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
            
       
            
       
        # pseudo_labeller = sla_no_uncertainty(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # SLA_noconfidence[tt]=pseudo_labeller.test_acc
        # if len(SLA_noconfidence[tt])<=NumIter:
        #     SLA_noconfidence[tt] = SLA_noconfidence[tt] + [SLA_noconfidence[tt][-1]]*(1+NumIter-len(SLA_noconfidence[tt]))
            
      
            
        # pseudo_labeller = UPS(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccUPS[tt]=pseudo_labeller.test_acc
        # if len(AccUPS[tt])<=NumIter:
        #     AccUPS[tt] = AccUPS[tt] + [AccUPS[tt][-1]]*(1+NumIter-len(AccUPS[tt]))
            
            
        # pseudo_labeller = lcb_ucb(copy.copy(xgb),
        #         x_unlabeled,x_test,y_test, 
        #         upper_threshold,lower_threshold,
        #         num_iters=NumIter,
        #         verbose = True
        #     )
        
        # pseudo_labeller.fit(x_train, y_train)
        # AccLCBUCB[tt]=pseudo_labeller.test_acc
        # if len(AccLCBUCB[tt])<=NumIter:
        #     AccLCBUCB[tt] = AccLCBUCB[tt] + [AccLCBUCB[tt][-1]]*(1+NumIter-len(AccLCBUCB[tt]))
            
       
      
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
        
        
        
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        # ax2.plot( np.asarray(pseudo_labeller.ce_difference_list)*100,'b-.',label="CE Difference")
        # ax2.set_ylabel("Student-Teacher Cross-Entropy",color='b')
        # ax2.tick_params(axis='y', labelcolor='b')
        
    # save result to file
    
    save_folder="results3"
    strFile=save_folder+"/flex_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccFlex)>0:
        np.save(strFile, AccFlex)
    
    strFile=save_folder+"/pl_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccPL)>0:
        np.save(strFile, AccPL)

    strFile=save_folder+"/Sinkhorn_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccSinkhorn)>0:
        np.save(strFile, AccSinkhorn)
        
    strFile=save_folder+"/CSA_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccCSA)>0:
        np.save(strFile, AccCSA)
    
    strFile=save_folder+"/CSA_TTest_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TTest)>0:
        np.save(strFile, CSA_TTest)
        
    strFile=save_folder+"/CSA_TotalVar_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TotalVar)>0:
        np.save(strFile, CSA_TotalVar)
        
    strFile=save_folder+"/CSA_TotalEnt_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TotalEnt)>0:
        np.save(strFile, CSA_TotalEnt)

    strFile=save_folder+"/CSA_TTest2_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TTest2)>0:
        np.save(strFile, CSA_TTest2)
        
    strFile=save_folder+"/CSA_TotalVar2_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TotalVar2)>0:
        np.save(strFile, CSA_TotalVar2)
        
    strFile=save_folder+"/CSA_TotalEnt2_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TotalEnt2)>0:
        np.save(strFile, CSA_TotalEnt2)
        
                
    strFile=save_folder+"/CSA_TotalVar_Ent_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(CSA_TotalVar_Ent)>0:
        np.save(strFile, CSA_TotalVar_Ent)
        
    strFile=save_folder+"/SLA_noconfidence_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(SLA_noconfidence)>0:
        np.save(strFile, SLA_noconfidence)
        
    strFile=save_folder+"/UPS_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccUPS)>0:
        np.save(strFile, AccUPS)
    
    strFile=save_folder+"/LCBUCB_{:d}_{:s}_{:d}_{:d}".format(ii,_datasetName[ii],FromIndex,ToIndex)
    if np.sum(AccLCBUCB)>0:
        np.save(strFile, AccLCBUCB)
    
    
    continue
