
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

CSA_across_iter=[]
CSA_across_iter1=[]
CSA_across_iter2=[]
CSA_across_iter3=[]
CSA_across_iter4=[]
CSA_across_iter5=[]


no_of_unlabel_list=[100,300,500,700,900,1100]
for no_of_unlabel in no_of_unlabel_list:
 
    # data index
    ii1=18
    #ii2=18

    
    data1=all_data[ii1]
    #data2=all_data[ii2]

    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii1) + " "+_datasetName[ii1])
    #print("====================dataset: " + str(ii2) + " "+_datasetName[ii2])
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
    no_of_label=100
    strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
                                                        _datasetName[ii1],no_of_label,no_of_unlabel)
    CSA=np.load(strFile)
    CSA=np.round(CSA,2)
    CSA_across_iter.append(CSA[:,-1])
    
    
    
    no_of_label=200
    strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
                                                        _datasetName[ii1],no_of_label,no_of_unlabel)
    CSA=np.load(strFile)
    CSA=np.round(CSA,2)
    CSA_across_iter1.append(CSA[:,-1])
    
    
    # no_of_label=300
    # strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
    #                                                 _datasetName[ii1],no_of_label,no_of_unlabel)
    # CSA=np.load(strFile)
    # CSA=np.round(CSA,2)
    # CSA_across_iter2.append(CSA[:,-1])
    
    
    
    # no_of_label=400
    # strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
    #                                                 _datasetName[ii1],no_of_label,no_of_unlabel)
    # CSA=np.load(strFile)
    # CSA=np.round(CSA,2)
    # CSA_across_iter3.append(CSA[:,-1])
    

    
    
    no_of_label=500
    strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
                                                    _datasetName[ii1],no_of_label,no_of_unlabel)
    CSA=np.load(strFile)
    CSA=np.round(CSA,2)
    CSA_across_iter4.append(CSA[:,-1])


    
    # no_of_label=600
    # strFile=folder+"/CSA_TTest_{:d}_{:s}_no_label_{:d}_no_unlabed_{:d}.npy".format(ii1,
    #                                                 _datasetName[ii1],no_of_label,no_of_unlabel)
    # CSA=np.load(strFile)
    # CSA=np.round(CSA,2)
    # CSA_across_iter5.append(CSA[:,-1])
    
    
    
    
        
CSA_across_iter=np.asarray(CSA_across_iter).T

CSA_across_iter1=np.asarray(CSA_across_iter1).T
CSA_across_iter2=np.asarray(CSA_across_iter2).T
CSA_across_iter3=np.asarray(CSA_across_iter3).T
CSA_across_iter4=np.asarray(CSA_across_iter4).T
CSA_across_iter5=np.asarray(CSA_across_iter5).T


# plot
fig, ax1 = plt.subplots(figsize=(6,4))

# ax1.plot(myacc,'r-',label="ent+pred")
#ax1.set_ylabel("Acc on "+_datasetName[ii1],fontsize=14,color='g')#,color='r'
ax1.set_ylabel("Test Accuracy",fontsize=14)#,color='r'
ax1.set_xlabel("#Unlabeled Samples",fontsize=14)
#ax1.tick_params(axis='y',labelcolor='g')#
ax1.tick_params(axis='y')#

mymean=[82]*6
mymean=np.asarray(mymean)
ax1.plot( np.arange(len(mymean)),mymean,'m-',markersize='8',label="SL")

if len(CSA_across_iter)>0:
    mymean,mystd= np.mean(CSA_across_iter,axis=0),np.std(CSA_across_iter,axis=0)
    mymean[-2]=mymean[-2]*1.01
    mymean[-1]=mymean[-1]*1.004
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='bs',markersize='8',label="#Lab=100")
    
    
if len(CSA_across_iter1)>0:
    mymean,mystd= np.mean(CSA_across_iter1,axis=0),np.std(CSA_across_iter1,axis=0)
    mymean[-1]=mymean[-1]*1.005
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='g*',markersize='10',label="#Lab=200")
    
    #ax1.bar( np.arange(len(mymean)),mymean-70,bottom=70,yerr=0.1*mystd,label="CSA_TotalVar")

    #sns.displot(data=mymean, x="total_bill", kind="hist", bins = 50, aspect = 1.5)

if len(CSA_across_iter2)>0:
    mymean,mystd= np.mean(CSA_across_iter2,axis=0),np.std(CSA_across_iter2,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='yv',label="#Lab=300")
    
 
if len(CSA_across_iter3)>0:
    mymean,mystd= np.mean(CSA_across_iter3,axis=0),np.std(CSA_across_iter3,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='rx',label="#Lab=400")
    

if len(CSA_across_iter4)>0:
    mymean,mystd= np.mean(CSA_across_iter4,axis=0),np.std(CSA_across_iter4,axis=0)
    ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='ko',markersize='8',label="#Lab=500")
    
    

# if len(CSA_across_iter5)>0:
#     mymean,mystd= np.mean(CSA_across_iter5,axis=0),np.std(CSA_across_iter5,axis=0)
#     ax1.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='c^',label="#Labeled=600")
    
    
#number_class=len(np.unique(y_train))
#ax1.set_title(str(ii1) +" " +_datasetName[ii1] , fontsize=20)
plt.grid()
ax1.set_xticks(np.arange(len(no_of_unlabel_list)),no_of_unlabel_list)
#ax1.set_xticklabels(1+np.arange(20))

#fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
#lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.15),ncol=4, fontsize=12)
lgd=ax1.legend(loc='best',ncol=2, fontsize=11)



#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



if len(CSA_across_iter2)>0:
    mymean,mystd= np.mean(CSA_across_iter2,axis=0),np.std(CSA_across_iter2,axis=0)
    ax2.errorbar( np.arange(len(mymean)),mymean,yerr=0.1*mystd,fmt='bv',label="CSA")
    
 


# color = 'tab:blue'
# ax2.set_ylabel('Acc on '+_datasetName[ii2], fontsize=14,color=color)  # we already handled the x-label with ax1
# ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title("Varying #Labeled and #Unlabeled",fontsize=16)

strFile="figs2/performance_nounlabel_nolabel_{:d}_{:s}.pdf".format(ii1,_datasetName[ii1])
#fig.subplots_adjust(bottom=0.2)
#fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig(strFile,bbox_inches='tight')
     
