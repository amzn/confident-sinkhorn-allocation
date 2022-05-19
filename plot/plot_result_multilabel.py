
import sys
  
# setting path
sys.path.append('..')
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.linear_model import SGDClassifier 
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn import preprocessing
#from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl
#from pseudo_labelling_algorithms import lcb_ucb
#from pseudo_labelling_algorithms import sla
from sklearn.preprocessing import StandardScaler
#from load_data import load_encode_data
import os
from load_multi_label_data import load_yeast_multilabel,load_emotions_multilabel,load_genbase_multilabel
from load_multi_label_data import load_corel5k_multilabel

from utils import rename_dataset
path ='../vector_data/'


_datasetName=[]
all_data=[]
data=load_yeast_multilabel(path) # binary
X=data['data']
Y = data['target']
_datasetName.append("breast_cancer")
#all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))
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

for ii, data in enumerate(all_data):
        
    # if ii==2:
    #     continue
    
    X = data['data']
    Y=data['target']
    
    # 30% train 70%test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # 25% train 25% unlabelled
    x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, 
                                                  test_size=0.5, random_state=0)
    
    
    print(_datasetName[ii])
    print(len(np.unique(y_test)))
    print("feat dim",x_train.shape[1])
    print("x_train.shape",x_train.shape)
    print("x_unlabeled.shape",x_unlabeled.shape)
    print("x_test.shape",x_test.shape)
    print("y_test",y_test.shape)


    mywidth=4

    # save result to file
    SLA=[]
    SLA_one=[]
    
    strFile="../multilabel_results/flex_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    print(strFile)
    #strFile="../multilabel_results/flex_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    AccFlex=np.load(strFile)[0]
    #HLFlex=np.load(strFile)[1]

    AccFlex=np.round(AccFlex,2)

    
    strFile="../multilabel_results/pl_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/pl_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    AccPL=np.load(strFile)[0]
    #HLPL=np.load(strFile)[1]

    AccPL=np.round(AccPL,2)


    strFile="multilabel_results/LCBUCB_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/LCBUCB_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    # AccLCBUCB=np.load(strFile)
    # AccLCBUCB=np.round(AccLCBUCB,2)
    
    strFile="multilabel_results/sinkhorn_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/sinkhorn_{:d}_{:s}.npy".format(ii,_datasetName[ii])

    #SLA=np.load(strFile)[0]
    #HLSLA=np.load(strFile,allow_pickle=True)[1]
    #SLA=np.round(SLA,2)
    
    try:  
        strFile="../multilabel_results/CSA_TotalEnt_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])

        CSA_TotalEnt=np.load(strFile)
        CSA_TotalEnt=np.round(CSA_TotalEnt,2)
    except:
        CSA_TotalEnt=[] 
        
        
        
    
    strFile="../multilabel_results/UPS_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/UPS_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    UPS=np.load(strFile)
    UPS=np.round(UPS,2)
    
    strFile="../multilabel_results/SLA_NoConfidence_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/SLA_NoConfidence_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    SLA_NoConfidence=np.load(strFile)
    SLA_NoConfidence=np.round(SLA_NoConfidence,2)
    # strFile="multilabel_results/SLA_Ori_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    # #strFile="multilabel_results/SLA_Ori_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    # SLA_ori=np.load(strFile)
    # SLA_ori=np.round(SLA_ori,2)
    
    strFile="../multilabel_results/CSA_TotalVar_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/CSA_TotalVar_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    CSA_TotalVar=np.load(strFile)
    CSA_TotalVar=np.round(CSA_TotalVar,2)
    CSA_TotalVar=np.asarray(CSA_TotalVar)
    
    strFile="../multilabel_results/CSA_TotalVar2_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    CSA_TotalVar2=np.load(strFile)
    CSA_TotalVar2=np.round(CSA_TotalVar2,2)
    CSA_TotalVar2=np.asarray(CSA_TotalVar2)
    
    
    strFile="../multilabel_results/CSA_TotalEnt2_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    CSA_TotalEnt2=np.load(strFile)
    CSA_TotalEnt2=np.round(CSA_TotalEnt2,2)
    CSA_TotalEnt2=np.asarray(CSA_TotalEnt2)
    
    
    strFile="../multilabel_results/CSA_TotalEnt_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    CSA_TotalEnt=np.load(strFile)
    CSA_TotalEnt=np.round(CSA_TotalEnt,2)
    CSA_TotalEnt=np.asarray(CSA_TotalEnt)
    
    
    strFile="../multilabel_results/CSA_TotalVar_Ent_{:d}_{:s}_0_15.npy".format(ii,_datasetName[ii])
    #strFile="../multilabel_results/CSA_TotalVar_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    CSA_TotalVar_Ent=np.load(strFile)
    CSA_TotalVar_Ent=np.round(CSA_TotalVar_Ent,2)
    CSA_TotalVar_Ent=np.asarray(CSA_TotalVar_Ent)
    
    # plot
    fig, ax1 = plt.subplots(figsize=(6,3.5))
    
    # ax1.plot(myacc,'r-',label="ent+pred")
    ax1.set_ylabel("Test Precision",fontsize=14)
    ax1.set_xlabel("Pseudo-label Iteration",fontsize=14)
    ax1.tick_params(axis='y')

    mymean,mystd= np.mean(AccPL,axis=0),np.std(AccPL,axis=0)
    x_axis=np.arange(len(mymean))
    
    
    supervised_learning_result=[ mymean[0] ]*len(x_axis)
    ax1.plot( x_axis,supervised_learning_result,'m:',linewidth=mywidth,label="Supervised")

    ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='k*--',elinewidth=mywidth,label="Pseudo Lab")

    mymean,mystd= np.mean(AccFlex,axis=0),np.std(AccFlex,axis=0)
    ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='gx--',elinewidth=mywidth,label="Flex")


    # mymean,mystd= np.mean(AccSinkhorn,axis=0),np.std(AccSinkhorn,axis=0)
    # ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='rs-',label="sinkhorn")

    # mymean,mystd= np.mean(AccSinkhorn2,axis=0),np.std(AccSinkhorn2,axis=0)
    # #ax1.plot(mymean,'c-',label="AccSinkhorn2")
    # ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='cv-',label="AccSinkhorn2")

    # if np.sum(AccLCBUCB)>0:
    #     mymean,mystd= np.mean(AccLCBUCB,axis=0),np.std(AccLCBUCB,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='yh:',label="U-LCB")

    if np.sum(UPS)>0:
        mymean,mystd= np.mean(UPS,axis=0),np.std(UPS,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='cv-',elinewidth=mywidth,label="UPS")

    # if np.sum(SLA_ori)>0:
    #     mymean,mystd= np.mean(SLA_ori,axis=0),np.std(SLA_ori,axis=0)
    #     ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='b<:',label="SLA")
    
    
    
    if np.sum(SLA_NoConfidence)>0:
        mymean,mystd= np.mean(SLA_NoConfidence,axis=0),np.std(SLA_NoConfidence,axis=0)
        ax1.errorbar( x_axis,mymean,yerr=0.1*mystd,fmt='bs:',elinewidth=mywidth,label="CSA No Conf")

    
    if len(CSA_TotalEnt)>0:
        mymean,mystd= np.mean(CSA_TotalEnt,axis=0),np.std(CSA_TotalEnt,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='b*-',label="CSA_TEnt")
    
    
    if len(CSA_TotalVar)>0:
        mymean,mystd= np.mean(CSA_TotalVar,axis=0),np.std(CSA_TotalVar_Ent,axis=0)
        ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,elinewidth=mywidth,fmt='rs-',label="CSA")
    

    # if np.sum(SLA)>0:
    #     mymean,mystd= np.mean(SLA,axis=0),np.std(SLA,axis=0)
    #     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rs-',label="CSA")
    
    if np.sum(CSA_TotalVar2)>0:
        mymean,mystd= np.mean(CSA_TotalVar2,axis=0),np.std(CSA_TotalVar2,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='kv--',label="CSA_TVar2")
        
        
    if np.sum(CSA_TotalEnt2)>0:
        mymean,mystd= np.mean(CSA_TotalEnt2,axis=0),np.std(CSA_TotalEnt2,axis=0)
        #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rv--',label="CSA_TEnt2")
        
        
    number_class=y_train.shape[1]
    #ax1.set_title(str(ii) +" " +_datasetName[ii] + " C=" + str(number_class),fontsize=20)
    dataset_name=rename_dataset(_datasetName[ii])
    ax1.set_title(dataset_name, fontsize=20)
    #fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
    #lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3)
    
    strFile="multilabel_figs/{:d}_{:s}_prec.pdf".format(ii,_datasetName[ii])
    #fig.subplots_adjust(bottom=0.2)
    #fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(strFile,bbox_inches='tight')
    
    
    
    