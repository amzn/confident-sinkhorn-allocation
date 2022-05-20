
import sys
import os
print(sys.path)
# setting path
sys.path.append('../')


print(sys.path)

#import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pseudo_labelling_algorithms import CSA, UPS

from pseudo_labelling_algorithms import Pseudo_Labeling,FlexMatch
from utils import get_train_test_unlabeled_data,rename_dataset

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
    
    if ii not in [6]:
        continue
        
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
    
    folder="../results4"
    #strFile=folder+"/flex_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    strFile=folder+"/flex_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccFlex=np.load(strFile)
    AccFlex=np.round(AccFlex,2)

    
    #strFile=folder+"/pl_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    strFile=folder+"/pl_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccPL=np.load(strFile)
    AccPL=np.round(AccPL,2)

    
    
    try:
        #strFile=folder+"/Sinkhorn_{:d}_{:s}_20_40.npy".format(ii,_datasetName[ii])
        strFile=folder+"/Sinkhorn_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/Sinkhorn_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        AccSinkhorn=np.load(strFile)
        AccSinkhorn=np.round(AccSinkhorn,2)
    except:
        AccSinkhorn=[]
    
        
       
    try:  
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar=np.load(strFile)
        CSA_TotalVar=np.round(CSA_TotalVar,2)
    except:
        CSA_TotalVar=[]  
       
    try:  
        strFile=folder+"/CSA_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA=np.load(strFile)
        CSA=np.round(CSA,2)
    except:
        CSA=[]  
        
        
    try:  
        strFile=folder+"/CSA_ttest_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_ttest_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_ttest=np.load(strFile)
        CSA_ttest=np.round(CSA_ttest,2)
    except:
        CSA_ttest=[]  
        
    try:  
        strFile=folder+"/CSA_TotalEnt_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        CSA_TotalEnt=np.load(strFile)
        CSA_TotalEnt=np.round(CSA_TotalEnt,2)
    except:
        CSA_TotalEnt=[] 


    try:  
        strFile=folder+"/CSA_NoConfidence_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        AccCSA_NoConfidence=np.load(strFile)
        AccCSA_NoConfidence=np.round(AccCSA_NoConfidence,2)
    except:
        AccCSA_NoConfidence=[] 
        
    # try:
    #     strFile=folder+"/Sinkhorn_ori_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    #     AccSinkhornOrig=np.load(strFile)
    #     AccSinkhornOrig=np.round(AccSinkhornOrig,2)
    # except:
    #     AccSinkhornOrig=[]

    
    try:
        strFile=folder+"/UPS_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
        UPS=np.load(strFile)
        UPS=np.round(UPS,2)
    except:
        UPS=[]
    
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
    
    
    if len(AccCSA_NoConfidence)>0:
        mymean,mystd= np.mean(AccCSA_NoConfidence,axis=0),np.std(AccCSA_NoConfidence,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='bs:',elinewidth=mywidth,label="SLA")

    if len(CSA)>0:
        mymean,mystd= np.mean(CSA,axis=0),np.std(CSA,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='y>-.',label="CSA prob>.5")
    

    if len(CSA_ttest)>0:
        mymean,mystd= np.mean(CSA_ttest,axis=0),np.std(CSA_ttest,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*-',label="CSA T-test")
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rs-',elinewidth=mywidth,label="CSA")
        print("CSA_Ttest {:.2f} ({:.1f})".format(np.mean(CSA_ttest[:,-1]),np.std(CSA_ttest[:,-1])))


    if len(CSA_TotalVar)>0:
        mymean,mystd= np.mean(CSA_TotalVar,axis=0),np.std(CSA_TotalVar,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='r*-',label="CSA TVar")
  
    
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
    
    #strFile="figs/{:d}_{:s}.pdf".format(ii,_datasetName[ii])
    strFile="{:d}_{:s}.pdf".format(ii,_datasetName[ii])
    
    fig.savefig(strFile,bbox_inches='tight')
        
    figlegend = plt.figure(figsize=(3,2))
    figlegend.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1],ncol=6, fontsize=10)
    strFile="figs/legend_pseudo_labeling.pdf"
    #figlegend.savefig(strFile, bbox_inches='tight')