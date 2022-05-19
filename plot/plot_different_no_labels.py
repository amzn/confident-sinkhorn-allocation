
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
from utils import get_train_test_unlabeled_data


path ='../vector_data/'


# Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/

def str2num(s, encoder):
    return encoder[s]




# load the data
with open('../all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)

CSA_across_iter1=[]
CSA_across_iter2=[]
    
    
no_of_label_list=[100,150,200,250,300,350,400,500]
for no_of_label in no_of_label_list:
 
    # data index
    ii1=10
    ii2=18

    
    data1=all_data[ii1]
    data2=all_data[ii2]

    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii1) + " "+_datasetName[ii1])
    print("====================dataset: " + str(ii2) + " "+_datasetName[ii2])
    #shapes.append(data.shape[0])

    # x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled_data(all_data, \
    #                                    _datasetName,dataset_index=ii,random_state=0)
        

    # print(_datasetName[ii])
    # print(len(np.unique(y_test)))
    # print("feat dim",x_train.shape[1])
    # print(x_train.shape)
    # print(x_unlabeled.shape)
    # print(x_test.shape)
    

        
    # save result to file
    
    folder="../results2"
    
    
    strFile=folder+"/CSA_TTest_{:d}_{:s}_0_20_no_label_{:d}.npy".format(ii1,_datasetName[ii1],no_of_label)
    CSA=np.load(strFile)
    CSA=np.round(CSA,2)
    CSA_across_iter1.append(CSA[:,-1])
    
    

    strFile=folder+"/CSA_TTest_{:d}_{:s}_0_20_no_label_{:d}.npy".format(ii2,_datasetName[ii2],no_of_label)
    CSA=np.load(strFile)
    CSA=np.round(CSA,2)
    CSA_across_iter2.append(CSA[:,-1])

    
        

CSA_across_iter1=np.asarray(CSA_across_iter1).T
CSA_across_iter2=np.asarray(CSA_across_iter2).T



# plot
fig, ax1 = plt.subplots(figsize=(6,4.5))

# ax1.plot(myacc,'r-',label="ent+pred")
ax1.set_ylabel("Acc on "+_datasetName[ii1],fontsize=14,color='g')#,color='r'
ax1.set_xlabel("#Labeled Samples",fontsize=14)
ax1.tick_params(axis='y',labelcolor='g')#

 
if len(CSA_across_iter1)>0:
    mymean,mystd= np.mean(CSA_across_iter1,axis=0),np.std(CSA_across_iter1,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='g*',label="CSA on Madelon")
    
    #ax1.bar( np.arange(len(mymean)),mymean-70,bottom=70,yerr=0.1*mystd,label="CSA_TotalVar")

    #sns.displot(data=mymean, x="total_bill", kind="hist", bins = 50, aspect = 1.5)


#number_class=len(np.unique(y_train))
#ax1.set_title(str(ii1) +" " +_datasetName[ii1] , fontsize=20)
plt.grid()
ax1.set_xticks(np.arange(len(no_of_label_list)),no_of_label_list)
#ax1.set_xticklabels(1+np.arange(20))

#fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
#lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


if len(CSA_across_iter2)>0:
    mymean,mystd= np.mean(CSA_across_iter2,axis=0),np.std(CSA_across_iter2,axis=0)
    ax2.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='kv',label="CSA on Digits")
    

#color = 'tab:blue'
color="black"
ax2.set_ylabel('Acc on '+_datasetName[ii2], fontsize=14,color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([80,95])
ax1.set_title("Performance by Varying #Labeled",fontsize=16)

ax1.legend()
ax2.legend(loc='lower right')

strFile="figs2/performance_wrt_nolabel_{:d}_{:s}.pdf".format(ii1,_datasetName[ii1])
#fig.subplots_adjust(bottom=0.2)
#fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig(strFile,bbox_inches='tight')
     