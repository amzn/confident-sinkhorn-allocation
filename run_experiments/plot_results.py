
import sys
import os
sys.path.append('../')


import numpy as np
import matplotlib.pyplot as plt

from confident_sinkhorn_allocation.algorithm.pseudo_labeling import Pseudo_Labeling
from confident_sinkhorn_allocation.algorithm.flexmatch import FlexMatch
from confident_sinkhorn_allocation.algorithm.ups import UPS
from confident_sinkhorn_allocation.algorithm.csa import CSA
from confident_sinkhorn_allocation.utilities.utils import get_train_test_unlabeled,get_train_test_unlabeled_for_multilabel

import pickle



# load the data
with open('all_data.pickle', 'rb') as handle:
    [all_data, _datasetName] = pickle.load(handle)


color_list=['k','b','y','r','g','c']
marker_list=['*','^','x','s','o','>']
linestyle_list=['--',':','-.','-']

save_dir = 'results_output' # path to the folder store the results
out_file='' 
numTrials=20 # number of repeated trials
numIters=5 # number of used pseudo-iterations

#====================================================================
# list of datasets 
#segment_2310_20 | wdbc_569_31 | analcatdata_authorship | synthetic_control_6c | \
        #German-credit |  madelon_no | agaricus-lepiota | breast_cancer | digits | emotions | yeast
dataset_name='synthetic_control_6c' 




list_algorithms=['Pseudo_Labeling','FlexMatch','UPS','CSA'] # list of algorithms to be plotted

# the following parameters to be used to load the correct paths
confidence='ttest' # for CSA 
upper_threshold=0.8 # for Pseudo_Labeling,FlexMatch
low_threshold=0.2 # for UPS
num_XGB_models=10 # for CSA and UPS

IsMultiLabel=False # by default
if dataset_name in ['yeast','emotions']: # multi-label
    IsMultiLabel=True


# load the data        
if IsMultiLabel==False: # multiclassification
            x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled(dataset_name,path_to_data='all_data.pickle',random_state=0)
else: # multi-label classification
    x_train,y_train, x_test, y_test, x_unlabeled=get_train_test_unlabeled_for_multilabel(dataset_name,path_to_data='all_data_multilabel.pickle',random_state=0)

    confidence='variance' # for CSA 





# %%
fig, ax1 = plt.subplots(figsize=(6,3.5))

ax1.set_ylabel("Test Accuracy",fontsize=14)
ax1.set_xlabel("Pseudo-label Iteration",fontsize=14)
ax1.tick_params(axis='y')



#Accuracy_List=[]
for idx,algo_name in enumerate(list_algorithms):

    if algo_name=='CSA':
        filename = os.path.join(save_dir, '{}_{}_{}_{}_M_{}_numIters_{}_numTrials_{}.pkl'.format(out_file, algo_name \
                ,confidence, dataset_name,num_XGB_models,numIters,numTrials))
    elif algo_name=='UPS':
        filename = os.path.join(save_dir, '{}_{}_{}_M_{}_numIters_{}_numTrials_{}_up_thresh_{}_low_thresh_{}.pkl'.format(out_file,\
             algo_name , dataset_name,num_XGB_models,numIters,numTrials,upper_threshold,low_threshold))
    else:
        filename = os.path.join(save_dir, '{}_{}_{}_numIters_{}_numTrials_{}_threshold_{}.pkl'.format(out_file, algo_name \
                , dataset_name,numIters,numTrials,upper_threshold))


    with open(filename, 'rb') as handle:
        accuracy = pickle.load(handle)

    #Accuracy_List.append(accuracy)

    accuracy = np.asarray(accuracy)
    accuracy=np.reshape(accuracy,(numTrials,numIters+1))

    mean,std= np.mean(accuracy,axis=0),np.std(accuracy,axis=0)
    x_axis=np.arange(len(mean))
    
    if idx==0:
        # supervised learning result is the first accuracy score in the list
        supervised_learning_result=[ mean[0] ]*len(x_axis)
        ax1.plot( np.arange(len(mean)),supervised_learning_result,'m:',linewidth=4,label="Supervised Learning")

    fmt=color_list[idx%len(color_list)]+marker_list[idx%len(marker_list)]+linestyle_list[idx%len(linestyle_list)]
    ax1.errorbar( np.arange(len(mean)),mean,yerr=0.1*std,fmt=fmt,elinewidth=4,label=algo_name)


number_class=len(np.unique(y_train))
ax1.set_title(dataset_name, fontsize=20)

plt.grid()

#fig.legend(loc='center')#legend(bbox_to_anchor=(1.1, 1.05))
lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=3, fontsize=12)

strFile="figs/{}.pdf".format(dataset_name)

fig.savefig(strFile,bbox_inches='tight')

print("====Saved the plot into " +strFile)
    

    