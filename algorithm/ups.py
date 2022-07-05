# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:19:22 2021

@author: Vu Nguyen
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
#from scipy.stats import entropy
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#import ot  # ot
from scipy import stats
import time					
from .pseudo_labeling import Pseudo_Labeling



# UPS: ===========================================================================================
#  Rizve, Mamshad Nayeem, Kevin Duarte, Yogesh S. Rawat, and Mubarak Shah. 
# "In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning." 
# ICLR. 2020.
#  https://arxiv.org/pdf/2101.06329.pdf
class UPS(Pseudo_Labeling):
    # adaptive thresholding
    
    def __init__(self,  unlabelled_data, x_test,y_test,num_iters=5,upper_threshold = 0.8, lower_threshold = 0.2,\
            num_XGB_models=10,verbose = False,IsMultiLabel=False):
        super().__init__( unlabelled_data, x_test,y_test,num_iters=num_iters,upper_threshold=upper_threshold,\
            lower_threshold=lower_threshold,num_XGB_models=num_XGB_models,verbose=verbose,IsMultiLabel=IsMultiLabel)
        
        self.algorithm_name="UPS"
      
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate_performance(self):
        super().evaluate_performance()
        
    def uncertainty_score(self, matrix_prob):
        return super().uncertainty_score(matrix_prob)

    def get_prob_at_max_class(self,pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)    
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def fit(self, X, y):
        
        print("=====",self.algorithm_name)

        self.nClass=self.get_number_of_labels(y)

        
        self.label_frequency=self.estimate_label_frequency(y)

        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y)
            self.evaluate_performance()
                                                
            # estimate prob using unlabelled data on M XGB models
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for mm in range(self.num_XGB_models):
                self.XGBmodels_list[mm].fit(X, y) # fit an XGB model
                pseudo_labels_prob_list[mm] = self.get_predictive_prob_for_unlabelled_data(self.XGBmodels_list[mm])
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
        
            #go over each row (data point), only keep the argmax prob
            # max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
        
            # # calculate uncertainty estimation for each data points at the argmax class
            # uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            # for ii in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
            #     idxMax=np.argmax( pseudo_labels_prob[ii,:] )
            #     uncertainty_rows[ii,idxMax]=np.std(pseudo_labels_prob_list[:,ii,idxMax])

            # augmented_idx=[]
            # MaxPseudoPoint=[0]*self.nClass
            # for cc in range(self.nClass):
            #     # compute the adaptive threshold for each class
               
            #     MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)

            #     idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
               
            #     idx_within_prob = np.where( max_prob_matrix[idx_sorted,cc] > self.upper_threshold )[0]
            #     idx_within_prob_uncertainty = np.where( uncertainty_rows[idx_sorted[idx_within_prob],cc] < self.lower_threshold)[0]
                
            #     labels_within_threshold=idx_sorted[idx_within_prob_uncertainty][:MaxPseudoPoint[cc]]
                
            #     augmented_idx += labels_within_threshold.tolist()

            #     X,y = self.post_processing(cc,labels_within_threshold,X,y)

            X,y=self.label_assignment_and_post_processing(pseudo_labels_prob,X,y,current_iter)
    
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return #self.test_acc
               
        
            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance() 
            
        #return self.test_acc