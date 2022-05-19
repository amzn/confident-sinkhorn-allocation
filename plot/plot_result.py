
import sys
  
# setting path
sys.path.append('..')
#import pandas as pd 
import numpy as np

#from sklearn.datasets import load_iris,load_breast_cancer,load_digits
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import copy

#from tqdm import tqdm
#from xgboost import XGBClassifier
#from sklearn import preprocessing
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import lcb_ucb
from pseudo_labelling_algorithms import csa
#from sklearn.preprocessing import StandardScaler
from utils import get_train_test_unlabeled_data,rename_dataset

#from load_data import load_encode_data
import os
import pickle

path ='../vector_data/'

def str2num(s, encoder):
    return encoder[s]

mywidth=4

# load the data
with open('../all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)

for ii, data in enumerate(all_data):
    
    # 
    #if ii in [0,9,11,12,13,14]:
    if ii in [0,1,9,11,12]:
        continue
    # if ii not in [12,13,14,15,16]:
    #     continue
    #if ii not in [1,2,4,11,12,13,14,15]:
    # if ii not in [16]:
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
    
    folder="../results"
    strFile=folder+"/flex_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    #strFile=folder+"/flex_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccFlex=np.load(strFile)
    AccFlex=np.round(AccFlex,2)

    
    strFile=folder+"/pl_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    #strFile=folder+"/pl_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccPL=np.load(strFile)
    AccPL=np.round(AccPL,2)

    
    # strFile="results/UncEntPred_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    # AccUncEntPred=np.load(strFile)
    # AccUncEntPred=np.round(AccUncEntPred,2)

    
    try:
        #strFile=folder+"/Sinkhorn_{:d}_{:s}_20_40.npy".format(ii,_datasetName[ii])
        strFile=folder+"/Sinkhorn_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        strFile=folder+"/Sinkhorn_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        AccSinkhorn=np.load(strFile)
        AccSinkhorn=np.round(AccSinkhorn,2)
    except:
        AccSinkhorn=[]
    
    try:  
        strFile=folder+"/SLA_noconfidence_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/SLA_noconfidence_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
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
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar=np.load(strFile)
        CSA_TotalVar=np.round(CSA_TotalVar,2)
    except:
        CSA_TotalVar=[]  
        
    try:  
        strFile=folder+"/CSA_TotalVar2_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_TotalVar2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar2=np.load(strFile)
        CSA_TotalVar2=np.round(CSA_TotalVar2,2)
    except:
        CSA_TotalVar2=[]  
        
    try:  
        strFile=folder+"/CSA_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/CSA_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA=np.load(strFile)
        CSA=np.round(CSA,2)
    except:
        CSA=[]  
        
        
    try:  
        strFile=folder+"/CSA_ttest_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/CSA_ttest_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_ttest=np.load(strFile)
        CSA_ttest=np.round(CSA_ttest,2)
    except:
        CSA_ttest=[]  
        
    try:  
        strFile=folder+"/CSA_ttest2_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_ttest2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_ttest2=np.load(strFile)
        CSA_ttest2=np.round(CSA_ttest2,2)
    except:
        CSA_ttest2=[]  
        
    try:  
        strFile=folder+"/CSA_TotalEnt_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        CSA_TotalEnt=np.load(strFile)
        CSA_TotalEnt=np.round(CSA_TotalEnt,2)
    except:
        CSA_TotalEnt=[] 


        
    try:  
        strFile=folder+"/CSA_TotalEnt2_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        CSA_TotalEnt2=np.load(strFile)
        CSA_TotalEnt2=np.round(CSA_TotalEnt2,2)
    except:
        CSA_TotalEnt2=[] 
    
        
    try:
        strFile=folder+"/Sinkhorn_ori_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        AccSinkhornOrig=np.load(strFile)
        AccSinkhornOrig=np.round(AccSinkhornOrig,2)
    except:
        AccSinkhornOrig=[]

    try:
        strFile=folder+"/LCBUCB_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        AccLCBUCB=np.load(strFile)
        AccLCBUCB=np.round(AccLCBUCB,2)
    except:
        AccLCBUCB=[]
    
    
    try:
        strFile=folder+"/UPS_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        UPS=np.load(strFile)
        UPS=np.round(UPS,2)
    except:
        UPS=[]
    
    # strFile="results/Sinkhorn_Label_Freq_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    # AccSinkhorn_LabelFreq=np.load(strFile)
    # #AccSinkhorn_LabelFreq=AccSinkhorn_LabelFreq[:10,:]
    # AccSinkhorn_LabelFreq=np.round(AccSinkhorn_LabelFreq,2)

    # plot
    fig, ax1 = plt.subplots(figsize=(6,3.5))
    
    # ax1.plot(myacc,'r-',label="ent+pred")
    ax1.set_ylabel("Test Accuracy",fontsize=14)#,color='r'
    ax1.set_xlabel("Pseudo-label Iteration",fontsize=14)
    ax1.tick_params(axis='y')#labelcolor='r'

    mymean,mystd= np.mean(AccPL,axis=0),np.std(AccPL,axis=0)
    x_axis=np.arange(len(mymean))
    
    
    supervised_learning_result=[ mymean[0] ]*len(x_axis)
    ax1.plot( np.arange(len(mymean)),supervised_learning_result,'m:',linewidth=mywidth,label="Supervised Learning")

    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*--',elinewidth=mywidth,label="Pseudo-Labeling")

    mymean,mystd= np.mean(AccFlex,axis=0),np.std(AccFlex,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='gx--',elinewidth=mywidth,label="Flex")


    if len(UPS)>0:
        mymean,mystd= np.mean(UPS,axis=0),np.std(UPS,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='cv-',elinewidth=mywidth,label="UPS")
    
    
    if len(SLA_noconfidence)>0:
        mymean,mystd= np.mean(SLA_noconfidence,axis=0),np.std(SLA_noconfidence,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='bs:',elinewidth=mywidth,label="SLA")


    if len(AccLCBUCB)>0:
        mymean,mystd= np.mean(AccLCBUCB,axis=0),np.std(AccLCBUCB,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='yh:',label="U-LCB")

    # if len(AccSinkhornOrig)>0:
    #     mymean,mystd= np.mean(AccSinkhornOrig,axis=0),np.std(AccSinkhornOrig,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b<:',label="SLA")
    
    if len(CSA)>0:
        mymean,mystd= np.mean(CSA,axis=0),np.std(CSA,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='y>-.',label="CSA prob>.5")
    

    if len(CSA_ttest)>0:
        mymean,mystd= np.mean(CSA_ttest,axis=0),np.std(CSA_ttest,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*-',label="CSA T-test")
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rs-',elinewidth=mywidth,label="CSA")
        print("CSA_Ttest {:.2f} ({:.1f})".format(np.mean(CSA_ttest[:,-1]),np.std(CSA_ttest[:,-1])))

    # if len(CSA_ttest2)>0:
    #     mymean,mystd= np.mean(CSA_ttest2,axis=0),np.std(CSA_ttest2,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='m*--',label="CSA T-test2")
    
    if len(CSA_TotalVar_Ent)>0:
        mymean,mystd= np.mean(CSA_TotalVar_Ent,axis=0),np.std(CSA_TotalVar_Ent,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='gv-',label="CSA_TVar_Ent")
    

    if len(CSA_TotalVar)>0:
        mymean,mystd= np.mean(CSA_TotalVar,axis=0),np.std(CSA_TotalVar,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='r*-',label="CSA TVar")
    
    # if len(CSA_TotalVar2)>0:
    #     mymean,mystd= np.mean(CSA_TotalVar2,axis=0),np.std(CSA_TotalVar2,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*--',label="CSA TVar2")
    
    if len(CSA_TotalEnt2)>0:
        mymean,mystd= np.mean(CSA_TotalEnt2,axis=0),np.std(CSA_TotalEnt2,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='y*--',label="CSA_TEnt2")
    
    
    if len(CSA_TotalEnt)>0:
        mymean,mystd= np.mean(CSA_TotalEnt,axis=0),np.std(CSA_TotalEnt,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b*-',label="CSA_TEnt")
    
    
    number_class=len(np.unique(y_train))
    dataset_name=rename_dataset(_datasetName[ii])
    ax1.set_title(dataset_name, fontsize=20)

    #ax1.set_title(str(ii) +" " +_datasetName[ii] + " K=" + str(number_class), fontsize=20)
    plt.grid()

    #fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
    #lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)
    
    strFile="figs/{:d}_{:s}.pdf".format(ii,_datasetName[ii])
    #fig.subplots_adjust(bottom=0.2)
    #fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(strFile,bbox_inches='tight')
        #import pandas as pd
        #pd.read_parquet('printer_train_AIWorkBench.parquet','fastparquet')
        
    figlegend = plt.figure(figsize=(3,2))
    figlegend.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1],ncol=6, fontsize=10)
    strFile="figs/legend_pseudo_labeling.pdf"
    figlegend.savefig(strFile, bbox_inches='tight')