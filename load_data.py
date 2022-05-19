# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:11:58 2021

@author: Vu Nguyen
"""
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
import pickle

import os

path ='./vector_data/'

#====================================================== read dataset

def load_encode_data( df ):
    
    x=df.values[:,:-1]
    y=df.values[:,-1]
    
    label_encoder = preprocessing.LabelEncoder()
    label_encoder = label_encoder.fit(y)
    label_encoded_y = label_encoder.transform(y)
    y=label_encoded_y
    
    
    features = []
    for i in range(0, x.shape[1]):
        
        try:
            x[:,i].astype(float)
            features.append(x[:,i])
        except:
        
            feat_encoder = preprocessing.LabelEncoder()
            feature = feat_encoder.fit_transform(x[:,i])
            features.append(feature)
    encoded_x = np.array(features).T
    encoded_x = encoded_x.reshape(x.shape[0], x.shape[1])
    
    x=encoded_x
    
    return x,y

#path_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data' # binary

#path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
#path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
#path_url='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'


#df = pd.read_csv(path_url)



#data = load_iris()
#data=load_breast_cancer() # binary
#data=load_digits()

#x=data['data']
#y = data['target']




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



#with open('all_data_protocol4.pickle', 'wb') as handle:
with open('all_data.pickle', 'wb') as handle:
    #pickle.dump([all_data,_datasetName], handle,protocol=4)
    pickle.dump([all_data,_datasetName], handle)
