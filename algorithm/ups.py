# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:19:22 2021

@author: Vu Nguyen
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
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

    def label_assignment_and_post_processing_UPS(self, pseudo_labels_prob,uncertainty_scores,X,y, current_iter=0,upper_threshold=None):
        """
        Given the threshold, we perform label assignment and post-processing

        Args:
            pseudo_labels_prob: predictive prob [N x K] where N is #unlabels, K is #class
            uncertainty_scores    : uncertainty_score of each data point at each class [N x K]
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

            
        assigned_pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass)).astype(int)

        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass): # loop over each class

            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
        
            idx_within_prob = np.where( max_prob_matrix[idx_sorted,cc] > self.upper_threshold )[0]
            idx_within_prob_uncertainty = np.where( uncertainty_scores[idx_sorted[idx_within_prob],cc] < self.lower_threshold)[0]
            
            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold=idx_sorted[idx_within_prob_uncertainty][:MaxPseudoPoint[cc]]
            
            assigned_pseudo_labels[labels_satisfied_threshold, cc]=1

        if self.verbose:
            print("MaxPseudoPoint",MaxPseudoPoint)
        
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
                                                
            # estimate prob using unlabelled data on M XGB models
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for mm in range(self.num_XGB_models):
                self.XGBmodels_list[mm].fit(X, y) # fit an XGB model
                pseudo_labels_prob_list[mm] = self.get_predictive_prob_for_unlabelled_data(self.XGBmodels_list[mm])
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
 
 
            # calculate uncertainty estimation for each data points at the argmax class
            uncertainty_scores=np.ones((pseudo_labels_prob.shape))
            for ii in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[ii,:] )
                uncertainty_scores[ii,idxMax]=np.std(pseudo_labels_prob_list[:,ii,idxMax])


            X,y=self.label_assignment_and_post_processing_UPS(pseudo_labels_prob,uncertainty_scores,X,y,current_iter)
    
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return
        
            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance() 
            