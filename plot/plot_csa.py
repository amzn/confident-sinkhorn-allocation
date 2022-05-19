
import sys
  
# setting path
sys.path.append('..')
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
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import lcb_ucb
from pseudo_labelling_algorithms import csa
from sklearn.preprocessing import StandardScaler
from utils import get_train_test_unlabeled_data

#from load_data import load_encode_data
import os
import pickle

path ='../vector_data/'

def str2num(s, encoder):
    return encoder[s]



# load the data
with open('../all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)

for ii, data in enumerate(all_data):
    
    # 
    #if ii in [0,9,11,12,13,14]:
    if ii in [0,1,9,11,12]:
        continue
    
    #if ii not in [1,2,4,11,12,13,14,15]:
    # if ii not in [17]:
    #     continue
        
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    #shapes.append(data.shape[0])

    x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled_data(all_data, \
                                       _datasetName,dataset_index=ii,random_state=0)
        
        

    print(_datasetName[ii])
    print(len(np.unique(y_test)))
    print("feat dim",x_train.shape[1])
    print("x_train.shape",x_train.shape)
    print("x_unlabeled.shape",x_unlabeled.shape)
    print("x_test.shape",x_test.shape)
    print("y_test",y_test.shape)



        
    # save result to file
    
    folder="../results3"
  
    
    # strFile="results/UncEntPred_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    # AccUncEntPred=np.load(strFile)
    # AccUncEntPred=np.round(AccUncEntPred,2)

 
    
    try:  
        strFile=folder+"/SLA_noconfidence_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/SLA_noconfidence_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        SLA_noconfidence=np.load(strFile)
        SLA_noconfidence=np.round(SLA_noconfidence,2)
    except:
        SLA_noconfidence=[]
        
    
    
    try:  
        strFile=folder+"/CSA_TotalVar_Ent_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_TotalVar_Ent_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        CSA_TotalVar_Ent=np.load(strFile)
        CSA_TotalVar_Ent=np.round(CSA_TotalVar_Ent,2)
    except:
        CSA_TotalVar_Ent=[]
        
       
    try:  
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar=np.load(strFile)
        CSA_TotalVar=np.round(CSA_TotalVar,2)
    except:
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])

        CSA_TotalVar=np.load(strFile)
        CSA_TotalVar=np.round(CSA_TotalVar,2)
        #CSA_TotalVar=[]
        
    try:  
        strFile=folder+"/CSA_TotalVar2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar2=np.load(strFile)
        CSA_TotalVar2=np.round(CSA_TotalVar2,2)
    except:
        strFile=folder+"/CSA_TotalVar2_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])

        CSA_TotalVar2=np.load(strFile)
        CSA_TotalVar2=np.round(CSA_TotalVar2,2)      
    # try:  
    #     strFile=folder+"/CSA_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    #     CSA=np.load(strFile)
    #     CSA=np.round(CSA,2)
    # except:
    #     strFile=folder+"/CSA_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])
    #     CSA=np.load(strFile)
    #     CSA=np.round(CSA,2) 
        
        
    try:  
        strFile=folder+"/CSA_ttest_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_ttest=np.load(strFile)
        CSA_ttest=np.round(CSA_ttest,2)
    except:
        strFile=folder+"/CSA_ttest_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])

        CSA_ttest=np.load(strFile)
        CSA_ttest=np.round(CSA_ttest,2)
        
    try:  
        strFile=folder+"/CSA_ttest2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_ttest2=np.load(strFile)
        CSA_ttest2=np.round(CSA_ttest2,2)
    except:
        strFile=folder+"/CSA_ttest2_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])

        CSA_ttest2=np.load(strFile)
        CSA_ttest2=np.round(CSA_ttest2,2)
    
    try:  
        strFile=folder+"/CSA_TotalEnt_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        
        CSA_TotalEnt=np.load(strFile)
        CSA_TotalEnt=np.round(CSA_TotalEnt,2)
    except:
        strFile=folder+"/CSA_TotalEnt_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])
        
        CSA_TotalEnt=np.load(strFile)
        CSA_TotalEnt=np.round(CSA_TotalEnt,2)
        
    try:  
        strFile=folder+"/CSA_TotalEnt2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        
        CSA_TotalEnt2=np.load(strFile)
        CSA_TotalEnt2=np.round(CSA_TotalEnt2,2)
    except:
        strFile=folder+"/CSA_TotalEnt2_{:d}_{:s}_0_10.npy".format(ii,_datasetName[ii])
        
        CSA_TotalEnt2=np.load(strFile)
        CSA_TotalEnt2=np.round(CSA_TotalEnt2,2)
    
  
    
    try:
        strFile=folder+"/UPS_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        UPS=np.load(strFile)
        UPS=np.round(UPS,2)
    except:
        UPS=[]
    

    # plot
    fig, ax1 = plt.subplots(figsize=(7,5))
    
    # ax1.plot(myacc,'r-',label="ent+pred")
    ax1.set_ylabel("Test Accuracy",fontsize=14)#,color='r'
    ax1.set_xlabel("Pseudo-label Iteration",fontsize=14)
    ax1.tick_params(axis='y')#labelcolor='r'

  

    if len(UPS)>0:
        mymean,mystd= np.mean(UPS,axis=0),np.std(UPS,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='cv--',label="UPS")
    
    # if len(AccSinkhornOrig)>0:
    #     mymean,mystd= np.mean(AccSinkhornOrig,axis=0),np.std(AccSinkhornOrig,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b<:',label="SLA")
    
    # if len(CSA)>0:
    #     mymean,mystd= np.mean(CSA,axis=0),np.std(CSA,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='c>-.',label="CSA prob>.5")
    

    if len(CSA_ttest)>0:
        mymean,mystd= np.mean(CSA_ttest,axis=0),np.std(CSA_ttest,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='r*-',label="CSA T-test")
    
    if len(CSA_ttest2)>0:
        mymean,mystd= np.mean(CSA_ttest2,axis=0),np.std(CSA_ttest2,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='r*--',label="CSA T-test2")
    
    # if len(CSA_TotalVar_Ent)>0:
    #     mymean,mystd= np.mean(CSA_TotalVar_Ent,axis=0),np.std(CSA_TotalVar_Ent,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='gv-',label="CSA_TVar_Ent")
    

    if len(CSA_TotalVar)>0:
        mymean,mystd= np.mean(CSA_TotalVar,axis=0),np.std(CSA_TotalVar,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*-',label="CSA TVar")
    
    if len(CSA_TotalVar2)>0:
        mymean,mystd= np.mean(CSA_TotalVar2,axis=0),np.std(CSA_TotalVar2,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*--',label="CSA TVar2")
    
     
    if len(CSA_TotalEnt)>0:
        mymean,mystd= np.mean(CSA_TotalEnt,axis=0),np.std(CSA_TotalEnt,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b*-',label="CSA_TEnt")
    
        print("CSA_TotalEnt {:.2f} ({:.1f})".format(np.mean(CSA_TotalEnt[:,-1]),np.std(CSA_TotalEnt[:,-1])))
    
    
    if len(CSA_TotalEnt2)>0:
        mymean,mystd= np.mean(CSA_TotalEnt2,axis=0),np.std(CSA_TotalEnt2,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b*--',label="CSA_TEnt2")
    
   
    number_class=len(np.unique(y_train))
    ax1.set_title(str(ii) +" " +_datasetName[ii] + " C=" + str(number_class), fontsize=20)
    plt.grid()

    #fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
    lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)
    
    strFile="figs3/{:d}_{:s}.pdf".format(ii,_datasetName[ii])
    #fig.subplots_adjust(bottom=0.2)
    fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
        #import pandas as pd
        #pd.read_parquet('printer_train_AIWorkBench.parquet','fastparquet')