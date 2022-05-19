
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
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled,rename_dataset

#from load_data import load_encode_data
import os
import pickle

path ='../vector_data/'

def str2num(s, encoder):
    return encoder[s]



# load the data
with open('../all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)

# plot
fig, ax1 = plt.subplots(figsize=(5.5,4))

# ax1.plot(myacc,'r-',label="ent+pred")
ax1.set_ylabel("Test Accuracy",fontsize=14)#,color='r'
ax1.set_xlabel("Pseudo-label Iteration",fontsize=14)
ax1.tick_params(axis='y')#labelcolor='r'
 

mywidth=4
 
threshold_list =[0.7,0.8,0.9,0.95]

for idx,upper_threshold in enumerate(threshold_list):
 
    ii=8
    
    
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    #shapes.append(data.shape[0])

    x_train,y_train, x_test, y_test, x_unlabeled=get_low_perf_train_test_unlabeled(all_data, \
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
    strFile=folder+"/flex_{:d}_{:s}_{:.2f}.npy".format(ii,_datasetName[ii],upper_threshold)
    #strFile=folder+"/flex_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccFlex=np.load(strFile)
    AccFlex=np.round(AccFlex,2)

    
    strFile=folder+"/pl_{:d}_{:s}_{:.2f}.npy".format(ii,_datasetName[ii],upper_threshold)
    #strFile=folder+"/pl_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])
    AccPL=np.load(strFile)
    AccPL=np.round(AccPL,2)

    
  
       
    try:  
        strFile=folder+"/CSA_TotalVar_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA_TotalVar=np.load(strFile)
        CSA_TotalVar=np.round(CSA_TotalVar,2)
    except:
        CSA_TotalVar=[]  
        
   
    try:  
        strFile=folder+"/CSA_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        #strFile=folder+"/CSA_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

        CSA=np.load(strFile)
        CSA=np.round(CSA,2)
    except:
        CSA=[]  
        
        
    
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
    
    
        


    color_list=['b','g','k','c','y']
   

    mymean,mystd= np.mean(AccPL,axis=0),np.std(AccPL,axis=0)
    
    
    
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt=color_list[idx]+'*-.',elinewidth=mywidth,label="$\gamma$=" + str(upper_threshold))

    mymean,mystd= np.mean(AccFlex,axis=0),np.std(AccFlex,axis=0)
    x_axis=np.arange(len(mymean))

    #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt=color_list[idx]+'x:',label="Flex $\gamma$=" + str(upper_threshold))

 




folder="../results"
try:  
    strFile=folder+"/CSA_ttest_{:d}_{:s}.npy".format(ii,_datasetName[ii])
    #strFile=folder+"/CSA_ttest_{:d}_{:s}_0_20.npy".format(ii,_datasetName[ii])

    CSA_ttest=np.load(strFile)
    CSA_ttest=np.round(CSA_ttest,2)
except:
    CSA_ttest=[]  
    


if len(CSA_ttest)>0:
    mymean,mystd= np.mean(CSA_ttest,axis=0),np.std(CSA_ttest,axis=0)
    #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*-',label="CSA T-test")
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rs-',elinewidth=mywidth,label="CSA")
    print("CSA_Ttest {:.2f} ({:.1f})".format(np.mean(CSA_ttest[:,-1]),np.std(CSA_ttest[:,-1])))

  
if len(CSA_TotalVar)>0:
    mymean,mystd= np.mean(CSA_TotalVar,axis=0),np.std(CSA_TotalVar,axis=0)
    #ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='r*-',label="CSA TVar")

    
# supervised_learning_result=[ mymean[0] ]*len(x_axis)
# ax1.plot( np.arange(len(mymean)),supervised_learning_result,'m:',linewidth=mywidth,label="SL")

    
number_class=len(np.unique(y_train))
dataset_name=rename_dataset(_datasetName[ii])
ax1.set_title(dataset_name, fontsize=20)
#ax1.set_title(str(ii) +" " +_datasetName[ii] + " K=" + str(number_class), fontsize=20)
plt.grid()

ax1.set_ylim([70.25,72.5])

fig.legend(loc='upper center',bbox_to_anchor=(.51, 0.9),fontsize=13.5,ncol=3)
#fig.legend(loc='center',bbox_to_anchor=(1.05, 0.5),fontsize=12)
#lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)

#fig.tight_layout()
strFile="figs/{:d}_{:s}_wrt_PL_threshold.pdf".format(ii,_datasetName[ii])
#fig.subplots_adjust(bottom=0.2)
#fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig(strFile,bbox_inches='tight')




        #import pandas as pd
        #pd.read_parquet('printer_train_AIWorkBench.parquet','fastparquet')
        
figlegend = plt.figure(figsize=(3,2))
figlegend.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1],ncol=6, fontsize=10)
strFile="figs/legend_PL_threshold.pdf"
#figlegend.savefig(strFile, bbox_inches='tight')