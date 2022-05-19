
import sys
  
# setting path
sys.path.append('..')

import numpy as np
#from sklearn.datasets import load_iris,load_breast_cancer,load_digits
import matplotlib.pyplot as plt
#from tqdm import tqdm
#from xgboost import XGBClassifier
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import lcb_ucb
from pseudo_labelling_algorithms import csa

#from load_data import load_encode_data

import pickle
from utils import get_train_test_unlabeled_data,get_low_perf_train_test_unlabeled,rename_dataset


path ='../vector_data/'


# Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/

def str2num(s, encoder):
    return encoder[s]


mywidth=3

# load the data
with open('../all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)

CSA_across_iter=[]
CSA_TotalVar_across_iter=[]
SupervisedLearning=[]
    
    
M_list =[1,2,3,5,7,10,12,15,20,25,30]
#M_list =[ 3,5,7,10,12,15,20]
for M in M_list:
 
    # data index
    ii=10
    
    data=all_data[ii]
        
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    #shapes.append(data.shape[0])

    x_train,y_train, x_test, y_test, x_unlabeled=get_low_perf_train_test_unlabeled(all_data, \
                                       _datasetName,dataset_index=ii,random_state=0)
        

    print(_datasetName[ii])
    print(len(np.unique(y_test)))
    print("feat dim",x_train.shape[1])
    print(x_train.shape)
    print(x_unlabeled.shape)
    print(x_test.shape)
    

        
    # save result to file
    
    folder="../results3"
    
    
    try:  
        #strFile=folder+"/CSA_{:d}_{:s}.npy".format(ii,_datasetName[ii])
        strFile=folder+"/CSA_TTest_{:d}_{:s}_M_{:d}.npy".format(ii,_datasetName[ii],M)
        CSA=np.load(strFile)
        CSA=np.round(CSA,2)
        SupervisedLearning.append(CSA[:,0])

        CSA_across_iter.append(CSA[:,-1])

    except:
        CSA=[]
    
    # try:  
    #     strFile=folder+"/CSA_TotalVar_{:d}_{:s}_0_20_Iter_{:d}.npy".format(ii,_datasetName[ii],NumIter)
        

    #     CSA_TotalVar=np.load(strFile)
    #     CSA_TotalVar=np.round(CSA_TotalVar,2)
    #     CSA_TotalVar_across_iter.append(CSA_TotalVar[:,-1])
    #     SupervisedLearning.append(CSA_TotalVar[:,0])
    # except:
    #     CSA=[]  
            
SupervisedLearning=np.asarray(SupervisedLearning).T
CSA_across_iter=np.asarray(CSA_across_iter).T
 



# plot
fig, ax1 = plt.subplots(figsize=(5,3.5))

# ax1.plot(myacc,'r-',label="ent+pred")
ax1.set_ylabel("Test Accuracy",fontsize=14)#,color='r'
ax1.set_xlabel("Number of XGB models M",fontsize=14)
ax1.tick_params(axis='y')#labelcolor='r'

 
# if len(CSA_across_iter)>0:
#     mymean,mystd= np.mean(CSA_across_iter,axis=0),np.std(CSA_across_iter,axis=0)
#     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='k*-',label="CSA")



if len(SupervisedLearning)>0:
    mymean,mystd= np.mean(SupervisedLearning,axis=0),np.std(SupervisedLearning,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='m*-',label="LSA")

if len(CSA_across_iter)>0:
    mymean,mystd= np.mean(CSA_across_iter,axis=0),np.std(CSA_across_iter,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='g*',\
             elinewidth=mywidth,markersize=5*mywidth,label="CSA")
    
    #ax1.bar( np.arange(len(mymean)),mymean-70,bottom=70,yerr=0.1*mystd,label="CSA_TotalVar")

    #sns.displot(data=mymean, x="total_bill", kind="hist", bins = 50, aspect = 1.5)


number_class=len(np.unique(y_train))

dataset_name=rename_dataset(_datasetName[ii])
#ax1.set_title(str(ii) +" " +_datasetName[ii] + " C=" + str(number_class), fontsize=20)
ax1.set_title(dataset_name, fontsize=18)
plt.grid()
ax1.set_xticks(np.arange(len(M_list)),M_list)
#ax1.set_xticklabels(1+np.arange(20))

#fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
#lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)
lgd=ax1.legend(loc='center',fancybox=True,ncol=1, fontsize=16)

strFile="figs2/perf_wrt_M_{:d}_{:s}.pdf".format(ii,_datasetName[ii])
#fig.subplots_adjust(bottom=0.2)
#fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig(strFile, bbox_inches='tight')
        #import pandas as pd
        #pd.read_parquet('printer_train_AIWorkBench.parquet','fastparquet')