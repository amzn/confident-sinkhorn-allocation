# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:19:22 2021

@author: Vu Nguyen
"""
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#import ot  # ot
from scipy import stats
import time					

    
class pseudo_labeling_iterative(object):
    # using all data points for pseudo-labels
    # iteratively take different percentage until all unlabelled data points are taken
    
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, 
                 lower_threshold = 0.2, num_iters=10,num_XGB_models=20,verbose = False,datasetName=None):
        
        self.name="pseudo_labeling_iterative"
        self.x_test=x_test
        self.y_test=y_test
        
        self.len_unlabels=[]
        self.len_accepted_ttest=[]
        self.len_selected=[]
       
        self.model = model
        self.teacher_model=model
        
        self.datasetName=datasetName
        
        self.unlabelled_data = unlabelled_data # this is a temporary unlabelled data changing in each iteration
        self.unlabelled_data_master = unlabelled_data # this is not changed
        self.verbose = verbose
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_iters=num_iters
        self.ce_difference_list=[]

        # allow the pseudo-data is repeated, e.g., without removing them after each iteration        
        # create a list of all the indices 
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))      
        
        self.selected_unlabelled_index=[]
        print("no of unlabelled data:",unlabelled_data.shape[0], "\t no of test data:",x_test.shape[0])

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)
        self.test_acc=[]
        self.ce_difference_list=[]
        self.FractionAllocatedLabel=1
        self.num_XGB_models=num_XGB_models

        # for uncertainty estimation
        # generate multiple models
        params = { 'max_depth': np.arange(3, 20).astype(int),
                   'learning_rate': [0.01, 0.1, 0.2, 0.3],
                   'subsample': np.arange(0.5, 1.0, 0.05),
                   'colsample_bytree': np.arange(0.4, 1.0, 0.05),
                   'colsample_bylevel': np.arange(0.4, 1.0, 0.05),
                   'n_estimators': [100, 200, 300, 500, 600, 700, 1000]}
        
        self.model_list=[0]*self.num_XGB_models
        
        param_list=[0]*self.num_XGB_models
        for tt in range(self.num_XGB_models):
            
            param_list[tt]={}
            
            for key in params.keys():
                #print(key)
                # select value
                
                mychoice=np.random.choice(params[key])
                #print(mychoice)
            
                param_list[tt][key]=mychoice
                param_list[tt]['verbosity'] = 0
                param_list[tt]['silent'] = 1
                param_list[tt]['seed'] = tt
                
            self.model_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)
    
    def estimate_label_frequency(self, y):
        # estimate the frequency
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        # smooth the label frequency if the ratio between the max class / min class is significant
        ratio=np.max(label_frequency)/np.min(label_frequency)
        if ratio>5:
            label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
    
        return label_frequency/np.sum(label_frequency)


    def data_uncertainty(self,pred):
    # pred [nModel, nPoint, nClass]
    
        ent=np.zeros((pred.shape[0],pred.shape[1]))
        for mm in range(pred.shape[0]):
            ent[mm,:]= self.entropy_prediction(pred[mm,:,:])
    
        return np.mean(ent,axis=0)

    def entropy_prediction(self,ave_pred,atClass=None):
        # pred [nPoint, nClass]
        
        ent=[0]*ave_pred.shape[0]
        
        for ii in range(ave_pred.shape[0]):
            ent[ii]= - np.sum( ave_pred[ii,:]*np.log(ave_pred[ii,:]))            
        return np.asarray(ent)
    
    def total_entropy(self,pred, atClass=None):
        ave_pred=np.mean(pred,axis=0) # average over model

        total_uncertainty=self.entropy_prediction(ave_pred,atClass)
        return total_uncertainty
    
    def knowledge_uncertainty(self,pred):
        
        total_uncertainty=self.total_uncertainty(pred)

        data_uncertainty=self.data_uncertainty(pred)

        knowledge_uncertainty = total_uncertainty-data_uncertainty
        return knowledge_uncertainty
    
    def total_variance(self,pred):
        # [nModel, nPoint, nClass]
        std_pred = np.std( pred, axis=0) # std over models
        total_std = np.sum(std_pred, axis=1) # sum of std over classes
        
        return total_std
    
    def calculate_ttest(self,pseudo_labels_prob_list):
        
        num_points=pseudo_labels_prob_list.shape[1]
        
        var_rows_argmax=[0]*num_points
        var_rows_arg2ndmax=[0]*num_points
        
        t_test=[0]*num_points
        t_value=[0]*num_points
        
        
        pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)

        temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
        idxargmax=temp[:,0]
        idx2nd_argmax= temp[:,1]

        for jj in range(num_points):# go over each row (data points)
        
            idxmax =idxargmax[jj]
            idx2ndmax=idx2nd_argmax[jj] 
            
            var_rows_argmax[jj]=np.var(pseudo_labels_prob_list[:,jj,idxmax  ])
            var_rows_arg2ndmax[jj]=np.var(pseudo_labels_prob_list[:,jj,idx2ndmax])
           
            nominator=pseudo_labels_prob[jj, idxmax]-pseudo_labels_prob[jj, idx2ndmax]
            temp=(0.1 + var_rows_argmax[jj] + var_rows_arg2ndmax[jj]  )/self.num_XGB_models
            denominator=np.sqrt(temp)
            t_test[jj] = nominator/denominator
            
            # compute degree of freedom
            nominator = (var_rows_argmax[jj] + var_rows_arg2ndmax[jj])**2
            
            denominator= var_rows_argmax[jj]**2 + var_rows_arg2ndmax[jj]**2
            denominator=denominator/(self.num_XGB_models-1)
            dof=nominator/denominator
        
            t_value[jj]=stats.t.ppf(1-0.025, dof)
            
            t_test[jj]=t_test[jj]-t_value[jj]
            
            
        
        return t_test
    
    
    def assignment_and_post_processing(self, assignment_matrix,X,y, 
                                           current_iter=0):
        
        num_points=assignment_matrix.shape[0]
        max_prob_matrix = self.get_prob_at_max_class(assignment_matrix)

        augmented_idx=[]
        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass):
            # compute the adaptive threshold for each class
            
            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
        
            #temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0]
            
            temp_idx1 = np.where( max_prob_matrix[idx_sorted,cc] > 0 )[0]
            #temp_idx2 = np.where( pseudo_labels_prob[idx_sorted[temp_idx1],cc] > 0.5)[0] # this is useless for binary classification
            
            labels_within_threshold=idx_sorted[temp_idx1][:MaxPseudoPoint[cc]]
            #labels_within_threshold =np.intersect1d( idx_sorted[temp_idx1] , idx_sorted[temp_idx2])
            
            #labels_within_threshold= idx_sorted[temp_idx1][:MaxPseudoPoint[cc]]
            
            augmented_idx += labels_within_threshold.tolist()

            X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
            
        print("MaxPseudoPoint",MaxPseudoPoint)

        if np.sum(self.num_augmented_per_class)==0:
            return X,y
        
        self.len_unlabels.append( len(self.unlabelled_data) )
        self.len_accepted_ttest.append( num_points ) 
        self.len_selected.append(  np.sum(self.num_augmented_per_class) )
        
    
    
        # remove the selected data from unlabelled data
        #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
        self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)  

        return X,y

    def assignment_and_post_processing_CSA(self, assignment_matrix,pseudo_labels_prob,X,y, 
                                           current_iter=0):
        
        num_points=assignment_matrix.shape[0]
        max_prob_matrix = self.get_prob_at_max_class(assignment_matrix)

        augmented_idx=[]
        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass):
            # compute the adaptive threshold for each class
            
            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
        
            #temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0]
            
            temp_idx1 = np.where( max_prob_matrix[idx_sorted,cc] > 0 )[0]
            temp_idx2 = np.where( pseudo_labels_prob[idx_sorted[temp_idx1],cc] > 0.5)[0] # this is useless for binary classification
            
            labels_within_threshold=idx_sorted[temp_idx2][:MaxPseudoPoint[cc]]
            #labels_within_threshold =np.intersect1d( idx_sorted[temp_idx1] , idx_sorted[temp_idx2])
            
            #labels_within_threshold= idx_sorted[temp_idx1][:MaxPseudoPoint[cc]]
            
            augmented_idx += labels_within_threshold.tolist()

            X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
            
        print("MaxPseudoPoint",MaxPseudoPoint)

        if np.sum(self.num_augmented_per_class)==0:
            return X,y
        
        self.len_unlabels.append( len(self.unlabelled_data) )
        self.len_accepted_ttest.append( num_points ) 
        self.len_selected.append(  np.sum(self.num_augmented_per_class) )
        
    
    
        # remove the selected data from unlabelled data
        #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
        self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)  

        return X,y
    
    def set_ot_regularizer(self,nRow,nCol):
        
        if nRow/nCol>=300:
            return 1
        if nRow/nCol>=200:
            return 0.5
        elif nRow/nCol>=100:
            return 0.2
        elif nRow/nCol>=50:
            return 0.1
        else:
            return 0.05
        
        
    # def uncertainty_score(self, matrix_prob):
        
    #     def KL(P,Q):
    #         """ Epsilon is used here to avoid conditional code for
    #         checking that neither P nor Q is equal to 0. """
    #         epsilon = 0.00001
        
    #         # You may want to instead make copies to avoid changing the np arrays.
    #         P = P+epsilon
    #         Q = Q+epsilon
       
    #         divergence = np.sum(P*np.log(P/Q))
    #         return divergence
 
    
    #     def cross_entropy(p, q):
    #         return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])
        
    #     # first compute the covariance matrix pairwise between row in matrix
    #     dist=np.zeros((matrix_prob.shape[0],matrix_prob.shape[0]))
    #     for ii in range(matrix_prob.shape[0]):
    #         for jj in range(matrix_prob.shape[0]):
    #             dist[ii,jj]=0.5* KL( matrix_prob[ii,:], matrix_prob[jj,:])+\
    #                 0.5*KL( matrix_prob[jj,:], matrix_prob[ii,:])
                
    #     # uncertainty is diversity in prediction
    #     covmatrix = np.exp( -dist)
        
    #     diversity=np.linalg.det(covmatrix)
    #     return diversity
                
        
    def cross_entropy_matrix(self, matrix_probA, matrix_probB):
        # elementwise between row in matrixA vs row in matrixB
        def cross_entropy(p, q):
            return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])
        
        if matrix_probA.shape[0] != matrix_probB.shape[0]:
            print("the number of rows in probA and probB is different")
            
        ce_value=[0]*matrix_probA.shape[0]
        for ii in range(matrix_probA.shape[0]):
            ce_value[ii]=cross_entropy( matrix_probA[ii,:], matrix_probB[ii,:])
        return np.average(ce_value)

    def evaluate(self):
        y_test_pred = self.model.predict(self.x_test)
        test_acc=accuracy_score(y_test_pred, self.y_test)*100

        print('+++Test Acc: {:.2f}%'.format(test_acc))
        self.test_acc +=[test_acc]
    
    def get_prob_at_max_class(self,pseudo_labels_prob):
        max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
        for jj in range(pseudo_labels_prob.shape[0]): 
            idxMax=np.argmax(pseudo_labels_prob[jj,:])
            max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
        return max_prob_matrix
    
    def post_processing(self,cc,labels_within_threshold,X,y, flagDelete=1):
        # Use argmax to find the class with the highest probability
        #pseudo_labels = np.argmax(pseudo_labels_prob[labels_within_threshold], axis = 1)
        pseudo_labels = [cc]*len(labels_within_threshold)
        pseudo_labels=np.asarray(pseudo_labels)
        chosen_unlabelled_rows = self.unlabelled_data[labels_within_threshold,:]

        self.selected_unlabelled_index += labels_within_threshold.tolist()
        
        self.num_augmented_per_class[cc]=len(pseudo_labels)
        
        
        if flagDelete:
            # remove the selected data from unlabelled data
            self.unlabelled_data = np.delete(self.unlabelled_data, labels_within_threshold, 0)

        # Combine data
        X = np.vstack((chosen_unlabelled_rows, X))
        y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))
        
        return X,y

    def get_max_pseudo_point(self,fraction_of_class, current_iter): # more at the begining and less at later stage
        #MaxPseudoPoint=fraction_of_class*np.int(self.FractionAllocatedLabel*len(self.unlabelled_data)/(self.num_iters*self.nClass))
        
        LinearRamp= [(self.num_iters-ii)/self.num_iters for ii in range(self.num_iters)]
        SumLinearRamp=np.sum(LinearRamp)
        
        fraction_iter= (self.num_iters-current_iter) / (self.num_iters*SumLinearRamp)
     
        MaxPseudoPoint=fraction_iter*fraction_of_class*self.FractionAllocatedLabel*len(self.unlabelled_data)

        return np.int(np.ceil(MaxPseudoPoint))


        
    def fit(self, X, y):
        """
        Perform pseudo labelling        
        X: train features
        y: train targets
        """
        print("===================",self.name)


        self.nClass=len(np.unique(y))
        
        
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")

        self.num_augmented_per_class=[0]*self.nClass

        self.label_frequency=self.estimate_label_frequency(y)

        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y.ravel())

            self.evaluate()
            
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            num_points = pseudo_labels_prob.shape[0]

            # remove the old augmented data
            #X= X[ np.sum(num_augmented_per_class):,:]
            #y= y[ np.sum(num_augmented_per_class):]

            #go over each row (data point), only keep the argmax prob
            max_prob=[0]*num_points
            
            max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            for jj in range(num_points): 
                idxMax=np.argmax(pseudo_labels_prob[jj,:])
                
                max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
                max_prob[jj]=pseudo_labels_prob[jj,idxMax]
        
            augmented_idx=[]
            for cc in range(self.nClass):

                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
    
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > self.upper_threshold)[0]
                labels_within_threshold= idx_sorted[temp_idx]
                    
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                augmented_idx += labels_within_threshold.tolist()

                
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
                
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
         
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()
        return self.test_acc
            
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
     
    def decision_function(self, X):
        return self.model.decision_function(X)

    
class flex_pl(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,verbose)
        self.name="flex"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        self.num_augmented_per_class=[0]*self.nClass
        
        self.label_frequency=self.estimate_label_frequency(y)


        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            num_points=pseudo_labels_prob.shape[0]
            # remove the old augmented data
        
            
            #go over each row (data point), only keep the argmax prob
            max_prob=[0]*num_points
            
            max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            for jj in range(num_points): 
                idxMax=np.argmax(pseudo_labels_prob[jj,:])
                
                max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
                max_prob[jj]=pseudo_labels_prob[jj,idxMax]
        
        
            # for each class, count the number of points > threshold
            countVector=[0]*self.nClass
            for cc in range(self.nClass):
                temp=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
                countVector[cc]= len( temp )
            countVector_normalized=np.asarray(countVector)/np.max(countVector)
            
            if self.verbose:
                print("class threshold:", np.round(countVector_normalized*self.upper_threshold,2))
            
            augmented_idx=[]
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                class_upper_thresh=countVector_normalized[cc]*self.upper_threshold


                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        

                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > class_upper_thresh)[0]
                labels_within_threshold= idx_sorted[temp_idx]
                augmented_idx += labels_within_threshold.tolist()

            
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
                
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)
                
            
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
          
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()
        return self.test_acc
    

class sinkhorn_original(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,verbose)
        self.name="sinkhorn_original"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)
    
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        
        self.num_augmented_per_class=[0]*self.nClass
        
        # estimate the frequency
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        #label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
    
            dist_points=np.ones( num_points+1 )
            
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=self.set_ot_regularizer(num_points,self.nClass)
            C=1-pseudo_labels_prob # cost # expand Cost matrix

            C=np.vstack((C,np.zeros((1,self.nClass))))
            C=np.hstack((C,np.zeros((num_points+1,1))))
            
            K=np.exp(-C/regulariser)
            
            
            # expand marginal dist for columns (class)
            #dist_labels=np.ones( pseudo_labels_prob.shape[1] )*(0.5*pseudo_labels_prob.shape[0])
            dist_labels = self.label_frequency*np.sum(dist_points)-1.0/self.nClass  # frequency of the class label
            dist_labels = np.append(dist_labels,1)
            
            
            if np.abs( np.sum(dist_labels) - np.sum(dist_points) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            uu=np.ones( (num_points+1,))
            
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= dist_labels / np.dot(K.T, uu)
                uu= dist_points / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            assignment_matrix=temp[:-1,:-1]
            #temp2 = ot.sinkhorn( dist_points , dist_labels, reg=regulariser, M=C )
            
            #pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
            #pseudo_labels_prob=temp2[:-1,:-1]

            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(assignment_matrix)

            augmented_idx=[]
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
                        
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx]
                augmented_idx += labels_within_threshold.tolist()
                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                
                
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
                
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   

                
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                        
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()  
        return self.test_acc
    

    
class sla_no_uncertainty(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,verbose)
        self.name="sinkhorn_label_assignment - no uncertainty"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)
            
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        self.num_augmented_per_class=[0]*self.nClass
        
        self.label_frequency=self.estimate_label_frequency(y)

        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]
            
            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            idxargmax=temp[:,0]
            idx2nd_argmax= temp[:,1]
            # get uncertainty estimation in argmax and 2nd argmax
            
            kappa=1
            
            # fraction of label being assigned.
            max_allocation_point= self.get_max_pseudo_point(class_freq=1,current_iter=current_iter)
            rho=max_allocation_point/ len(self.unlabelled_data)
            rho=1
            
            row_marginal=np.ones(num_points)

            upper_b_per_class=self.label_frequency*1.1
            lower_b_per_class=self.label_frequency*0.9
            
            temp=num_points*rho*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            row_marginal = np.append(row_marginal,temp)
            
            regulariser=self.set_ot_regularizer(num_points, self.nClass)
            
            
            print("#unlabel={:d} #points/#classes={:d}/{:d}={:.2f} reg={:.2f}".format(
                len(self.unlabelled_data),num_points,self.nClass,num_points/self.nClass,regulariser))


            C=1-pseudo_labels_prob # cost # expand Cost matrix

            C=np.vstack((C,np.zeros((1,self.nClass))))
            C=np.hstack((C,np.zeros((num_points+1,1))))
            
            K=np.exp(-C/regulariser)
            
            # expand marginal dist for columns (class)
            #col_marginal = upper_b_per_class*np.sum(t_test[idxNoneZero])  # frequency of the class label
            col_marginal = rho*upper_b_per_class*num_points  # frequency of the class label

            #temp=np.sum(t_test[idxNoneZero])*(1-np.sum(lower_b_per_class))
            temp=num_points*(1-rho*np.sum(lower_b_per_class))

            col_marginal = np.append(col_marginal,temp)
            
            
            if np.abs( np.sum(col_marginal) - np.sum(row_marginal) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            uu=np.ones( (num_points+1,))
            
            for jj in range(100):
                vv= col_marginal / np.dot(K.T, uu)
                uu= row_marginal / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            #temp2 = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
            
            #pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
            assignment_matrix=temp[:-1,:-1]
            
            
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(assignment_matrix)

            augmented_idx=[]
            MaxPseudoPoint=[0]*self.nClass
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                
                MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
            
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0]
                labels_within_threshold= idx_sorted[temp_idx][:MaxPseudoPoint[cc]]
                
                augmented_idx += labels_within_threshold.tolist()

                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                
            print("MaxPseudoPoint",MaxPseudoPoint)

            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            
                        
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()  
        return self.test_acc
    
class csa(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.9, 
                 lower_threshold = 0.1, num_iters=10,num_XGB_models=20,confidence_choice="ttest",verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,num_XGB_models,verbose)
        
        self.confidence_choice=confidence_choice
        self.name="csa  + confidence_choice:" + confidence_choice
        
        self.elapse_xgb=[]
        self.elapse_ttest=[]
        self.elapse_sinkhorn=[]
        
        print("M=",self.num_XGB_models)
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)
    def total_uncertainty(self,pred):
        return super().total_uncertainty(pred)
            
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        self.num_augmented_per_class=[0]*self.nClass
        
        self.label_frequency=self.estimate_label_frequency(y)

        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]
            pseudo_labels_prob_list=[0]*self.num_XGB_models

            tic = time.perf_counter() 

            # estimate prob using unlabelled data            
            for tt in range(self.num_XGB_models):
                self.model_list[tt].fit(X, y.ravel())
                pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
            toc = time.perf_counter() 
            self.elapse_xgb.append(toc-tic)

            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
         
            #print("pseudo_labels_prob_list.shape",pseudo_labels_prob_list.shape)
            # get argmax in each row and second best 2nd argmax in each row
            #idxargmax = np.argmax(pseudo_labels_prob_list, axis=0)
            temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            
            
            tic = time.perf_counter() # Start Time

            
            # estimate confidence level here====================================
            if self.confidence_choice=="variance":
                tot_variance=self.total_variance(pseudo_labels_prob_list)
                confidence=1-tot_variance
                confidence=confidence-np.mean(confidence)
            elif self.confidence_choice=="neg_variance":
                confidence=self.total_variance(pseudo_labels_prob_list)
                confidence=confidence-np.mean(confidence)
            elif self.confidence_choice=='entropy':
                tot_ent=self.total_entropy(pseudo_labels_prob_list)
                confidence=1-tot_ent
                confidence=confidence-0.5*np.mean(confidence)
            elif self.confidence_choice=='neg_entropy':
                confidence=self.total_entropy(pseudo_labels_prob_list)
                confidence=confidence-np.mean(confidence)
                
            elif self.confidence_choice=="ttest":
                confidence=self.calculate_ttest(pseudo_labels_prob_list)
            elif self.confidence_choice=="neg_ttest":
                confidence=self.calculate_ttest(pseudo_labels_prob_list)
                confidence=-np.asarray(confidence)
            elif self.confidence_choice=="no": 
                confidence=np.ones((1,num_points))
                
            confidence=np.clip(confidence, a_min=0,a_max=np.max(confidence))
            
            toc = time.perf_counter() # End Time
            self.elapse_ttest.append(toc-tic)
            
            # for numerical stability of OT, select the nonzero entry only
            idxNoneZero=np.where( confidence>0 )[0]
            #idxNoneZero=np.where( (confidence>0) & (confidence<0.9*np.max(confidence)) )[0]
            num_points= len(idxNoneZero)
            if len(idxNoneZero)==0: # terminate if could not find any point satisfying constraints
                return self.test_acc
            
            
            # fraction of label being assigned.
            max_allocation_point= self.get_max_pseudo_point(class_freq=1,current_iter=current_iter)
            rho=max_allocation_point/ len(self.unlabelled_data)
            
            
            tic = time.perf_counter() 



            #row_marginal=t_test[idxNoneZero]
            row_marginal=np.ones(num_points)
            
            upper_b_per_class=self.label_frequency*1.1
            lower_b_per_class=self.label_frequency*0.9
            
            #temp=np.sum(t_test[idxNoneZero])*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            temp=num_points*rho*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            row_marginal = np.append(row_marginal,temp)
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=self.set_ot_regularizer(num_points, self.nClass)
            
            
            print("#unlabel={:d} #points/#classes={:d}/{:d}={:.2f} reg={:.2f}".format(
                len(self.unlabelled_data),num_points,self.nClass,num_points/self.nClass,regulariser))
            
            
            C=1-pseudo_labels_prob # cost # expand Cost matrix
            C=C[idxNoneZero,:]
            
            C=np.vstack((C,np.zeros((1,self.nClass))))
            C=np.hstack((C,np.zeros((len(idxNoneZero)+1,1))))
            
            K=np.exp(-C/regulariser)
            
            # expand marginal dist for columns (class)
            #col_marginal = upper_b_per_class*np.sum(t_test[idxNoneZero])  # frequency of the class label
            col_marginal = rho*upper_b_per_class*num_points  # frequency of the class label
            
            #temp=np.sum(t_test[idxNoneZero])*(1-np.sum(lower_b_per_class))
            temp=num_points*(1-rho*np.sum(lower_b_per_class))
            
            col_marginal = np.append(col_marginal,temp)
            
            
            if np.abs( np.sum(col_marginal) - np.sum(row_marginal) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            uu=np.ones( (num_points+1,))
            
            for jj in range(100):
                vv= col_marginal / np.dot(K.T, uu)
                uu= row_marginal / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            #temp2 = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
            
            
            toc = time.perf_counter() 
            self.elapse_sinkhorn.append(toc-tic)



            assignment_matrix=np.zeros((pseudo_labels_prob.shape))
            assignment_matrix[idxNoneZero,:]=temp[:-1,:-1]
            
            X,y=self.assignment_and_post_processing(assignment_matrix,X,y, 
                                                    current_iter) 
            
            # X,y=self.assignment_and_post_processing_CSA(assignment_matrix,pseudo_labels_prob,X,y, 
            #                                         current_iter) 
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            
                        
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()  
        return self.test_acc
    


class lcb_ucb(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,\
                         num_iters,verbose)
        self.name="lcb_ucb"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        self.num_augmented_per_class=[0]*self.nClass

        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        self.label_frequency=label_frequency/np.sum(label_frequency)
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            #pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            
            for tt in range(self.num_XGB_models):
                self.model_list[tt].fit(X, y.ravel())
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
            #vu=np.mean(pseudo_labels_prob_list,axis=0) # debug
        
            #print("pseudo_labels_prob_list.shape",pseudo_labels_prob_list.shape)
            # get argmax in each row and second best 2nd argmax in each row
            #idxargmax = np.argmax(pseudo_labels_prob_list, axis=0)
            temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            idxargmax=temp[:,0]
            idx2nd_argmax= temp[:,1]
            # get uncertainty estimation in argmax and 2nd argmax
            
            kappa=1
                        
            # calculate uncertainty estimation for each data points===============
            #uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            uncertainty_rows_argmax=[0]*num_points
            uncertainty_rows_arg2ndmax=[0]*num_points
            
            lcb_ucb=np.zeros((num_points,self.nClass))
            for jj in range(num_points):# go over each row (data points)
            
                idxmax =idxargmax[jj]
                idx2ndmax=idx2nd_argmax[jj] 
                
                uncertainty_rows_argmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idxmax  ])
                uncertainty_rows_arg2ndmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idx2ndmax])
                
                # lcb of argmax - ucb of arg2ndmax
                lcb_ucb[jj,idxmax] = ( pseudo_labels_prob[jj, idxmax] - kappa*uncertainty_rows_argmax[jj] ) \
                                    - (pseudo_labels_prob[jj, idx2ndmax] + kappa*uncertainty_rows_arg2ndmax[jj])
            
          
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(lcb_ucb)
            augmented_idx=[]
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)

                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
            
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx][:MaxPseudoPoint]
                augmented_idx += labels_within_threshold.tolist()
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
            
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   

            
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            
                        
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()  
        return self.test_acc
    
    
    

#1. generate multiple XGB with different hyper

    

# UPS: https://arxiv.org/pdf/2101.06329.pdf
class UPS(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, \
                 num_iters=10,verbose = False, datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,\
                         num_iters,verbose,datasetName)
        
        self.name="UPS"

      
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
        
    def uncertainty_score(self, matrix_prob):
        return super().uncertainty_score(matrix_prob)

    def get_prob_at_max_class(self,pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)    
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def fit(self, X, y):
        
        print("====================",self.name)

        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
                    
        self.num_augmented_per_class=[0]*self.nClass
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        self.label_frequency=label_frequency/np.sum(label_frequency)

        #print("self.lower_threshold",self.lower_threshold)
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            self.evaluate()

            for tt in range(self.num_XGB_models):
                self.model_list[tt].fit(X, y.ravel())
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
        
            #go over each row (data point), only keep the argmax prob
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
            
         
            # calculate uncertainty estimation for each data points
            uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                uncertainty_rows[jj,idxMax]=np.std(pseudo_labels_prob_list[:,jj,idxMax])
                #uncertainty_rows[jj,idxMax]=self.uncertainty_score(pseudo_labels_prob_list[:,jj,:])
       
            # larger is better

            augmented_idx=[]
            MaxPseudoPoint=[0]*self.nClass
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
               
                MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)

                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
               
                temp_idx1 = np.where( max_prob_matrix[idx_sorted,cc] > self.upper_threshold )[0]
                temp_idx2 = np.where( uncertainty_rows[idx_sorted[temp_idx1],cc] < self.lower_threshold)[0]
                #labels_within_threshold =np.intersect1d( idx_sorted[temp_idx1] , idx_sorted[temp_idx2])
                labels_within_threshold=idx_sorted[temp_idx2][:MaxPseudoPoint[cc]]
                
                augmented_idx += labels_within_threshold.tolist()

                X,y = self.post_processing(cc,labels_within_threshold,X,y,flagDelete=0)
                
            print("MaxPseudoPoint",MaxPseudoPoint)
                
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
               
                    
            # remove the selected data from unlabelled data
            #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   


            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate() 
            
        return self.test_acc
    
    
    