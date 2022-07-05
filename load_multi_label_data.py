# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:25:12 2022

@author: Vu Nguyen
"""


#import arff
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

def load_yeast_multilabel(folder=''):
    # temp = arff.loadarff(open('vector_data/yeast-train.arff', 'r'))
    # df_train = pd.DataFrame(temp[0])
    
    # temp = arff.loadarff(open('vector_data/yeast-test.arff', 'r'))
    # df_test = pd.DataFrame(temp[0])
    
    # X_train=df_train.values[:,:103]
    # Y_train=df_train.values[:,103:].astype(int)
    
    
    # X_test=df_test.values[:,:103]
    # Y_test=df_test.values[:,103:].astype(int)
    
    temp = arff.loadarff(open(folder+'yeast.arff', 'r'))
    df = pd.DataFrame(temp[0])
    
    data={}
    data['target']=df.values[:,103:].astype(int)
    data['data']=df.values[:,:103]
    
    return data




def load_emotions_multilabel(folder=''):
    # temp = arff.loadarff(open('vector_data/yeast-train.arff', 'r'))
    # df_train = pd.DataFrame(temp[0])
    
    # temp = arff.loadarff(open('vector_data/yeast-test.arff', 'r'))
    # df_test = pd.DataFrame(temp[0])
    
    # X_train=df_train.values[:,:103]
    # Y_train=df_train.values[:,103:].astype(int)
    
    
    # X_test=df_test.values[:,:103]
    # Y_test=df_test.values[:,103:].astype(int)
    
    temp = arff.loadarff(open(folder+'emotions/emotions.arff', 'r'))
    df = pd.DataFrame(temp[0])
    
    data={}
    data['target']=df.values[:,-6:].astype(int)
    data['data']=df.values[:,:-6]
    
    return data




def load_genbase_multilabel(folder=''):
    temp = arff.loadarff(open(folder+'/genbase/genbase.arff', 'r'))
    df = pd.DataFrame(temp[0])
    
    data={}
    data['target']=df.values[:,-27:].astype(int)
    data['data']=df.values[:,:-27]
    
    ord_enc = OneHotEncoder()
    data['data'] = ord_enc.fit_transform(data['data'])
    data['data']=data['data'].todense()

    
    return data


def load_corel5k_multilabel(folder):
    temp = arff.loadarff(open(folder+'corel5k/corel5k.arff', 'r'))
    df = pd.DataFrame(temp[0])
    
    data={}
    data['target']=df.values[:,-374:].astype(int)
    data['data']=df.values[:,:-374]
    
    # ord_enc = OneHotEncoder()
    # data['data'] = ord_enc.fit_transform(data['data'])
    # data['data']=data['data'].todense()

    
    return data

#vu=load_emotions_multilabel()
#vu=load_corel5k_multilabel()
path ='./vector_data/'

all_data=[]
_datasetName=['emotions','yeast','genbase']
all_data.append(load_yeast_multilabel(path))
all_data.append(load_emotions_multilabel(path))
all_data.append(load_genbase_multilabel(path))

with open('all_data_multilabel.pickle', 'wb') as handle:
    pickle.dump([all_data,_datasetName], handle)