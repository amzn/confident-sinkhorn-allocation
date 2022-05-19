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
from pseudo_labelling_algorithms import pseudo_labeling_iterative,flex_pl,entropy_pl,prediction_entropy
#from pseudo_labelling_algorithms import flex_pl_teacher_student,pl_iterative_teacher_student
from pseudo_labelling_algorithms import uncertainty_prediction,uncertainty_entropy_score,lcb_ucb
from pseudo_labelling_algorithms import uncertainty_pl,uncertainty_entropy,sinkhorn_label_assignment,sinkhorn_label_assignment_2
from sklearn.preprocessing import StandardScaler
from load_data import load_encode_data
import os

path ='./vector_data/'


# Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/

def str2num(s, encoder):
    return encoder[s]



param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['verbosity'] = 0
param['silent'] = 1
#     param["eval_metric"] = "error"
#     param['eta'] = 0.5
#     param['gamma'] = 0.2
#     param['max_depth'] = 3
#     param['min_child_weight']=1
#     param['max_delta_step'] = 0
#     param['subsample']= 0.5
#     param['colsample_bytree']=1
#     
#     param['seed'] = 0
#     param['base_score'] = 0.5


# create XGBoost instance with default hyper-parameters
#xgb = XGBClassifier(objective='binary:logistic',verbosity = 0)
xgb = XGBClassifier(**param,use_label_encoder=False)

log_reg = SGDClassifier(loss = 'log', n_jobs = -1, alpha = 1e-5)







all_data = []
s_dataname = []
shapes = []

_datasetName =["cjs","hill-valley","segment_2310_20","wdbc_569_31","steel-plates-fault",
          "analcatdata_authorship","synthetic_control_6c","vehicle_846_19","German-credit",
          "gina_agnostic_no","madelon_no","texture","gas_drift","dna_no"
           ] #
for ii in range(len(_datasetName)):
    temp = pd.read_csv(os.path.join(path ,_datasetName[ii]+".csv"))
    if temp.shape[0] <= 30000000:
        print(_datasetName[ii])
        all_data.append(temp)
        s_dataname.append(_datasetName[ii]+".csv")
        
        
# load data from UCI
path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("car")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

    
path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("kr_vs_kp")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
X,Y=load_encode_data( pd.read_csv(path_url) )
_datasetName.append("agaricus-lepiota")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))




# data = load_iris()
# X=data['data']
# Y = data['target']
# _datasetName.append("iris")
# all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))

data=load_breast_cancer() # binary
X=data['data']
Y = data['target']
_datasetName.append("breast_cancer")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))


data=load_digits()
X=data['data']
Y = data['target']
_datasetName.append("digits")
all_data.append( np.hstack((X, np.reshape(Y,(-1,1)))))



for ii, data in enumerate(all_data):
 
    # if ii in [0,2,5,7,11]:
    #     continue
    
    if ii in [0]:
        continue
    
    # if ii!= 14:
    #     continue
    
    ## import the data
    #data = all_data[data_num]
    print("====================dataset: " + str(ii) + " "+_datasetName[ii])
    shapes.append(data.shape[0])

    if ii<14:
        _dic = list(set(data.values[:, -1]))
        num_labels = len(_dic)
        encoder = {}
        for i in range(len(_dic)):
            encoder[_dic[i]] = i
    
        # shuffle original dataset
        data = data.sample(frac=1)
        X = data.values[:, :-1]
        # X = scale(X)  # scale the X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = np.array([str2num(s, encoder) for s in data.values[:, -1]])
    else:
        X = data[:, :-1]
        Y=data[:,-1]
        
    
    # 30% train 70%test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    
    # 25% train 25% unlabelled
    x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, test_size=0.6, random_state=42)
    
    p = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[p], y_train[p]
    
    p = np.random.permutation(x_unlabeled.shape[0])
    x_unlabeled, y_unlabeled = x_unlabeled[p], y_unlabeled[p]
    
    fig, ax1 = plt.subplots()

    NumIter=11
    upper_threshold=0.8
    lower_threshold=0.15

    # pseudo_labeling_iterative
    # flex_pl
    # flex_pl_teacher_student
    # entropy_pl
    # pl_teacher_student_stop
    
    # method 1    
    pseudo_labeller = pseudo_labeling_iterative(copy.copy(xgb),
            x_unlabeled,x_test,y_test, # for evaluation purpose
            upper_threshold,lower_threshold,
            num_iters=NumIter,verbose = True
        )
    
    pseudo_labeller.fit(x_train, y_train)
    myacc=pseudo_labeller.test_acc
    if len(myacc)<NumIter:
        myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    ax1.plot(myacc,'k-',label="pseudo labelling")
    
    
    # method 2    
    pseudo_labeller = flex_pl(copy.copy(xgb),
            x_unlabeled,x_test,
            y_test,upper_threshold, # for evaluation purpose
            num_iters=NumIter,verbose = True
        )
    
    pseudo_labeller.fit(x_train, y_train)
    myacc=pseudo_labeller.test_acc
    if len(myacc)<NumIter:
        myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    ax1.plot(myacc,'kx:',label="FLEX")
  
    
    # # method 3    
    # pseudo_labeller = entropy_pl(copy.copy(xgb),
    #         x_unlabeled,x_test,y_test, # for evaluation purpose
    #         num_iters=NumIter,verbose = True
    #     )
    
    # pseudo_labeller.fit(x_train, y_train)
    # myacc=pseudo_labeller.test_acc
    # if len(myacc)<NumIter:
    #     myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    # ax1.plot(myacc,'k^-',label="entropy")
    
    # method 3    
    # pseudo_labeller = uncertainty_prediction(copy.copy(xgb),
    #         x_unlabeled,x_test,y_test, upper_threshold,lower_threshold,# for evaluation purpose
    #         num_iters=NumIter,verbose = True
    #     )
    
    # pseudo_labeller.fit(x_train, y_train)
    # myacc=pseudo_labeller.test_acc
    # if len(myacc)<NumIter:
    #     myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    # ax1.plot(myacc,'gs-',label="uncert+pred")
 
    
    #method 1
    # pseudo_labeller = prediction_entropy(
    #         copy.copy(xgb),
    #         x_unlabeled,x_test,
    #         y_test, upper_threshold,lower_threshold,# for evaluation purpose
    #         num_iters=NumIter,
    #         flagRepetition=False,
    #         verbose = True
    #     )
    
    # pseudo_labeller.fit(x_train, y_train)
    # myacc=pseudo_labeller.test_acc
    # if len(myacc)<NumIter:
    #     myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    # ax1.plot(myacc,'r-',label="ent+pred")
    ax1.set_ylabel("Test Accuracy",color='r')
    ax1.set_xlabel("Pseudo-label Iteration")
    ax1.tick_params(axis='y', labelcolor='r')
    
    #method 4   
    pseudo_labeller = uncertainty_entropy_score(copy.copy(xgb),
            x_unlabeled,x_test,
            y_test, upper_threshold,lower_threshold,# for evaluation purpose
            num_iters=NumIter,
            flagRepetition=False,
            verbose = True,datasetName=_datasetName[ii],
        )
    
    pseudo_labeller.fit(x_train, y_train)
    myacc=pseudo_labeller.test_acc
    if len(myacc)<NumIter:
        myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))

    ax1.plot(myacc,'mo-',label="uncert+ent+pred")
    
    #method 4   
    pseudo_labeller = sinkhorn_label_assignment(copy.copy(xgb),
            x_unlabeled,x_test,y_test, 
            upper_threshold,lower_threshold,
            num_iters=NumIter,
            flagRepetition=False,
            verbose = True
        )
    
    pseudo_labeller.fit(x_train, y_train)
    myacc=pseudo_labeller.test_acc
    if len(myacc)<NumIter:
        myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    ax1.plot(myacc,'b-',label="sinkhorn_label_assignment")
    
    #method 5   
    pseudo_labeller = lcb_ucb(copy.copy(xgb),
            x_unlabeled,x_test,y_test, 
            upper_threshold,lower_threshold,
            num_iters=NumIter,
            flagRepetition=False,
            verbose = True
        )
    
    pseudo_labeller.fit(x_train, y_train)
    myacc=pseudo_labeller.test_acc
    if len(myacc)<NumIter:
        myacc = myacc + [myacc[-1]]*(NumIter-len(myacc))
        
    ax1.plot(myacc,'c-',label="lcb_ucb")
    
    #method 5   
    # pseudo_labeller = pl_iterative_teacher_student(copy.copy(xgb),
    #         x_unlabeled,x_test,
    #         y_test, # for evaluation purpose
    #         num_iters=NumIter,
    #         flagRepetition=False,
    #         verbose = False
    #     )
    
    # pseudo_labeller.fit(x_train, y_train)
    # myacc=pseudo_labeller.test_acc

    # ax1.plot(myacc,'c:',label="teacher-student")
    
    
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # ax2.plot( np.asarray(pseudo_labeller.ce_difference_list)*100,'b-.',label="CE Difference")
    # ax2.set_ylabel("Student-Teacher Cross-Entropy",color='b')
    # ax2.tick_params(axis='y', labelcolor='b')
    
    number_class=len(np.unique(y_train))
    ax1.set_title('Dataset ' + str(ii) +" " +_datasetName[ii] + " C=" + str(number_class))

    #fig.legend(loc='center')legend(bbox_to_anchor=(1.1, 1.05))
    lgd=ax1.legend(loc='upper center',fancybox=True,bbox_to_anchor=(0.5, -0.2),ncol=2)
    
    strFile="figs5/{:s}.pdf".format(_datasetName[ii])
    #fig.subplots_adjust(bottom=0.2)
    fig.savefig(strFile,bbox_extra_artists=(lgd,), bbox_inches='tight')
    #import pandas as pd
    #pd.read_parquet('printer_train_AIWorkBench.parquet','fastparquet')