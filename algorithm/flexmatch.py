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




# FlexMatch Strategy for Pseudo-Labeling =======================================================================
# Zhang, Bowen, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. 
# "Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling." NeurIPS 2021
class FlexMatch(Pseudo_Labeling):
    # adaptive thresholding
    
    def __init__(self, unlabelled_data, x_test,y_test,num_iters=5,upper_threshold = 0.9, verbose = False,IsMultiLabel=False):
        """
        unlabelled_data      : [N x d] where N is the number of unlabeled data, d is the feature dimension
        x_test               :[N_test x d]
        y_test               :[N_test x 1] for multiclassification or [N_test x K] for multilabel classification
        num_iters            : number of pseudo-iterations, recommended = 5 as in the paper
        upper_threshold      : the upper threshold used for pseudo-labeling, e.g., we assign label if the prob > 0.8
        fraction_allocation  : the faction of label allocation, if fraction_allocation=1, we assign labels to 100% of unlabeled data
        lower_threshold      : lower threshold, used for UPS 
        num_XGB_models       : number of XGB models used for UPS and CSA, recommended = 10
        verbose              : verbose
        IsMultiLabel         : False => Multiclassification or True => Multilabel classification
        """

        super().__init__( unlabelled_data, x_test,y_test,num_iters=num_iters,upper_threshold=upper_threshold,verbose=verbose,IsMultiLabel=IsMultiLabel)
        
        self.algorithm_name="FlexMatch"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate_performance(self):
        super().evaluate_performance()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)

    def label_assignment_and_post_processing_FlexMatch(self, pseudo_labels_prob,X,y, current_iter=0,upper_threshold=None):
        """
        Given the threshold, perform label assignments and augmentation
        This function is particular for FlexMatch
        Args:
            pseudo_labels_prob: predictive prob [N x K] where N is #unlabels, K is #class
            X: existing pseudo_labeled + labeled data [ N' x d ]
            y: existing pseudo_labeled + labeled data [ N' x 1 ] for multiclassification
            y: existing pseudo_labeled + labeled data [ N' x K ] for multilabel classification

        Output:
            Augmented X = augmented_X + X
            Augmented y = augmented_y + Y
        """
        
    if self.IsMultiLabel==False:
        #go over each row (data point), only keep the argmax prob 
        # because we only allow a single data point to a single class
        max_prob_matrix=self.get_prob_at_max_class(pseudo_labels_prob)
    else:
        # we dont need to get prob at max class for multi-label
        # because a single data point can be assigned to multiple classes
        max_prob_matrix=pseudo_labels_prob


    # for each class, count the number of points > threshold
    # this is the technique used in FlexMatch
    countVector=[0]*self.nClass
    for cc in range(self.nClass):
        temp=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
        countVector[cc]= len( temp )
    countVector_normalized=np.asarray(countVector)/np.max(countVector)
    

    if upper_threshold is None:
        upper_threshold=self.upper_threshold
        

    # assign labels if the prob > threshold ========================================================
    assigned_pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass)).astype(int)
    MaxPseudoPoint=[0]*self.nClass
    for cc in range(self.nClass): # loop over each class

        # note that in FlexMatch, the upper_threshold is updated below before using as the threshold
        flex_class_upper_thresh=countVector_normalized[cc]*self.upper_threshold

        # obtain the maximum number of points can be assigned per class
        MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
        
        idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        

        temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > flex_class_upper_thresh )[0]   
        labels_satisfied_threshold=idx_sorted[temp_idx]

        # only select upto MaxPseudoPoint[cc] points
        labels_satisfied_threshold = labels_satisfied_threshold[:MaxPseudoPoint[cc]] 
        assigned_pseudo_labels[labels_satisfied_threshold, cc]=1


    if self.verbose:
        print("MaxPseudoPoint",MaxPseudoPoint)
    
    # post-processing and augmenting the data into X and Y ==========================================
    return self.post_processing_and_augmentation(assigned_pseudo_labels,X,y)



    def fit(self, X, y):
        """
        main algorithm to perform pseudo labelling     

        Args:   
            X: train features [N x d]
            y: train targets [N x 1]

        Output:
            we record the test_accuracy a vector of test accuracy per pseudo-iteration
        """

        print("=====",self.algorithm_name)
        
        self.nClass=self.get_number_of_labels(y)

        
        self.label_frequency=self.estimate_label_frequency(y)

        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y)
            
            self.evaluate_performance()

            # estimate prob using unlabelled data
            pseudo_labels_prob=self.get_predictive_prob_for_unlabelled_data(self.model)

            
            #go over each row (data point), only keep the argmax prob
            # max_prob=[0]*num_points
            
            # max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            # for ii in range(num_points): 
            #     idxMax=np.argmax(pseudo_labels_prob[ii,:])
                
            #     max_prob_matrix[ii,idxMax]=pseudo_labels_prob[ii,idxMax]
            #     max_prob[ii]=pseudo_labels_prob[ii,idxMax]
        
        
            # for each class, count the number of points > threshold
            # countVector=[0]*self.nClass
            # for cc in range(self.nClass):
            #     idx_above_threshold=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
            #     countVector[cc]= len( idx_above_threshold ) # count number of unlabeled data above the threshold
            # countVector_normalized=np.asarray(countVector)/np.max(countVector)
            
            # if self.verbose:
            #     print("class threshold:", np.round(countVector_normalized*self.upper_threshold,2))
            
            X,y=self.label_assignment_and_post_processing_FlexMatch( pseudo_labels_prob,X,y, current_iter=0)

            # augmented_idx=[]
            # for cc in range(self.nClass):
            #     # compute the adaptive threshold for each class
            #     class_upper_thresh=countVector_normalized[cc]*self.upper_threshold

            #     MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            #     idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        

            #     idx_above_threshold = np.where(max_prob_matrix[idx_sorted,cc] > class_upper_thresh)[0]
            #     labels_within_threshold= idx_sorted[idx_above_threshold]
            #     augmented_idx += labels_within_threshold.tolist()

            #     X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
                
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
          

            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return #self.test_acc
                
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)
                
        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance()
    

    