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
import ot  # ot

# class pseudo_labeling_iterative():
#     # using all data points for pseudo-labels
#     # iteratively take different percentage until all unlabelled data points are taken
    
    
#     def __init__(self, model, unlabelled_data, x_test_poly,y_test,
#                  upper_threshold = 0.6, lower_threshold = 0.4, num_iters=10,verbose = False):
        
#         self.x_test=x_test_poly
#         self.y_test=y_test
       
#         self.model = model
#         self.unlabelled_data = unlabelled_data
#         self.verbose = verbose
#         self.upper_threshold = upper_threshold
#         self.lower_threshold = lower_threshold
#         self.num_iters=num_iters
#         # create a list of all the indices 
#         self.unlabelled_indices = list(range(unlabelled_data.shape[0]))      
#         print(unlabelled_data.shape[0])

#         # these are for large dataset
#         #sample_rate=0.1
#         #self.sample_rate = sample_rate

#         # Number of rows to sample in each iteration
#         #self.sample_size = int(unlabelled_data.shape[0] * self.sample_rate)
        
#         #if num_iter is None:
#             #self.num_iters = int(len(self.unlabelled_indices)/self.sample_size)

#         #print(self.sample_size)
#         #print(self.sample_rate)

#         # Shuffle the indices
#         np.random.shuffle(self.unlabelled_indices)


#     def __pop_rows(self):
#         """
#         Function to sample indices without replacement
#         """
#         #chosen_rows = self.unlabelled_indices[:self.sample_size]
#         chosen_rows = self.unlabelled_indices
        
#         # Remove the chosen rows from the list of indicies (We are sampling w/o replacement)
#         #self.unlabelled_indices = self.unlabelled_indices[self.sample_size:]
#         self.unlabelled_indices = self.unlabelled_indices[self.sample_size:]
#         return chosen_rows
    
    
#     def fit(self, X, y):
        
#         """
#         Perform pseudo labelling
        
#         X: train features
#         y: train targets
        
#         """
        

#         for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
#             # Get the samples
#             chosen_rows = self.__pop_rows()

#             # Fit to data
#             self.model.fit(X, y.ravel())

#             chosen_unlabelled_rows = self.unlabelled_data[chosen_rows,:]
#             pseudo_labels_prob = self.model.predict_proba(chosen_unlabelled_rows)
            
            
#             # We have 10 classes this means `predict_proba` returns an array of 10 probabilities per datapoint
#             # We will first find the maximum probability and then find the rows which are within our threshold values
#             label_probability = np.max(pseudo_labels_prob, axis = 1)
#             labels_within_threshold = np.where((label_probability < self.lower_threshold) | 
#                                                (label_probability > self.upper_threshold))[0]
            
            
#            # Use argmax to find the class with the highest probability
#             pseudo_labels = np.argmax(pseudo_labels_prob[labels_within_threshold], axis = 1)
#             chosen_unlabelled_rows = chosen_unlabelled_rows[labels_within_threshold]

#             # Combine data
#             X = np.vstack((chosen_unlabelled_rows, X))
#             y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))

#             # Shuffle 
#             indices = list(range(X.shape[0]))
#             np.random.shuffle(indices)

#             X = X[indices]
#             y = y[indices]     
            
#             y_test_pred = self.predict(self.x_test)
#             print('Test Accuracy: {:.2f}%'.format(accuracy_score(y_test_pred, self.y_test)*100))
        
#     def predict(self, X):
#         return self.model.predict(X)
    
#     def predict_proba(self, X):
#         return self.model.predict_proba(X)
    
     
#     def decision_function(self, X):
#         return self.model.decision_function(X)
    
    
class pseudo_labeling_iterative(object):
    # using all data points for pseudo-labels
    # iteratively take different percentage until all unlabelled data points are taken
    
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, 
                 lower_threshold = 0.2, num_iters=10,flagRepetition=True,verbose = False,datasetName=None):
        
        self.name="pseudo_labeling_iterative"
        self.x_test=x_test
        self.y_test=y_test
       
        self.model = model
        self.teacher_model=model
        
        self.datasetName=datasetName
        
        self.unlabelled_data = unlabelled_data # this is a temporary unlabelled data changing in each iteration
        self.unlabelled_data_master = unlabelled_data # this is not changed
        self.verbose = verbose
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_iters=num_iters

        # allow the pseudo-data is repeated, e.g., without removing them after each iteration        
        self.flagRepetition=flagRepetition
        # create a list of all the indices 
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))      
        
        self.selected_unlabelled_index=[]
        print("no of unlabelled data:",unlabelled_data.shape[0], "\t no of test data:",x_test.shape[0])

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)
        self.test_acc=[]
        self.ce_difference_list=[]
        self.MaxPseudoPoint_per_Class=20
        self.FractionAllocatedLabel=0.5
        self.num_XGB_models=20

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
                
                mychoice=random.choice(params[key])
                #print(mychoice)
            
                param_list[tt][key]=mychoice
                param_list[tt]['verbosity'] = 0
                param_list[tt]['silent'] = 1
                param_list[tt]['seed'] = tt
                
            self.model_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)


    def uncertainty_score(self, matrix_prob):
        
        def KL(P,Q):
            """ Epsilon is used here to avoid conditional code for
            checking that neither P nor Q is equal to 0. """
            epsilon = 0.00001
        
            # You may want to instead make copies to avoid changing the np arrays.
            P = P+epsilon
            Q = Q+epsilon
       
            divergence = np.sum(P*np.log(P/Q))
            return divergence
 
    
        def cross_entropy(p, q):
            return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])
        
        # first compute the covariance matrix pairwise between row in matrix
        dist=np.zeros((matrix_prob.shape[0],matrix_prob.shape[0]))
        for ii in range(matrix_prob.shape[0]):
            for jj in range(matrix_prob.shape[0]):
                dist[ii,jj]=0.5* KL( matrix_prob[ii,:], matrix_prob[jj,:])+\
                    0.5*KL( matrix_prob[jj,:], matrix_prob[ii,:])
                
        # uncertainty is diversity in prediction
        covmatrix = np.exp( -dist)
        
        diversity=np.linalg.det(covmatrix)
        return diversity
                
        
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

        print('Test Acc: {:.2f}%'.format(test_acc))
        self.test_acc +=[test_acc]
    
    def get_prob_at_max_class(self,pseudo_labels_prob):
        max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
        for jj in range(pseudo_labels_prob.shape[0]): 
            idxMax=np.argmax(pseudo_labels_prob[jj,:])
            max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
        return max_prob_matrix
    
    def post_processing(self,cc,labels_within_threshold,X,y):
        # Use argmax to find the class with the highest probability
        #pseudo_labels = np.argmax(pseudo_labels_prob[labels_within_threshold], axis = 1)
        pseudo_labels = [cc]*len(labels_within_threshold)
        pseudo_labels=np.asarray(pseudo_labels)
        chosen_unlabelled_rows = self.unlabelled_data[labels_within_threshold,:]

        self.selected_unlabelled_index += labels_within_threshold.tolist()
        
        self.num_augmented_per_class[cc]=len(pseudo_labels)

        # Combine data
        X = np.vstack((chosen_unlabelled_rows, X))
        y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))
        
        # remove the selected data from unlabelled data
        self.unlabelled_data = np.delete(self.unlabelled_data, labels_within_threshold, 0)
        
        return X,y
    
    def get_max_pseudo_point(self):
        MaxPseudoPoint=np.int(self.FractionAllocatedLabel*len(self.unlabelled_data)/(self.num_iters*self.nClass))
        return MaxPseudoPoint

        
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

        
        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
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
        
            
            for cc in range(self.nClass):

                MaxPseudoPoint=self.get_max_pseudo_point()
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
    
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > self.upper_threshold)[0]
                labels_within_threshold= idx_sorted[temp_idx]
                    
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)

                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc

                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                
                
            # Shuffle 
            # indices = list(range(X.shape[0]))
            # np.random.shuffle(indices)

            # X = X[indices]
            # y = y[indices]     
        return self.test_acc
            
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
     
    def decision_function(self, X):
        return self.model.decision_function(X)

# class pl_iterative_teacher_student(pseudo_labeling_iterative):
#     # adaptive thresholding
    
#     def __init__(self, model, unlabelled_data, x_test,y_test,
#                  upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
#         super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
#     def predict(self, X):
#         super().predict(X)
#     def predict_proba(self, X):
#         super().predict_proba(X)
#     def evaluate(self):
#         super().evaluate()
#     def cross_entropy_matrix(self,probA, probB):
#         return super().cross_entropy_matrix(probA, probB)
#     def post_processing(self,cc,labels_within_threshold,X,y):
#         return super().post_processing(cc,labels_within_threshold,X,y)
    
#     def fit(self, X, y):
        
#         self.nClass=len(np.unique(y))
#         if len(np.unique(y)) < len(np.unique(self.y_test)):
#             print("num class in training data is less than test data !!!")
            
#         self.num_augmented_per_class=[0]*self.nClass

#         self.teacher_model.fit(X, y.ravel())

#         for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

#             # Fit to data
#             self.model.fit(X, y.ravel())
            
#             self.evaluate()

#             # estimate prob using unlabelled data
#             pseudo_labels_prob_student = self.model.predict_proba(self.unlabelled_data)
#             pseudo_labels_prob_teacher = self.teacher_model.predict_proba(self.unlabelled_data)
            
#             pseudo_labels_prob=0.5*pseudo_labels_prob_student+0.5*pseudo_labels_prob_teacher
            
#             ce_difference=self.cross_entropy_matrix(pseudo_labels_prob_teacher, pseudo_labels_prob_student)
#             self.ce_difference_list += [ce_difference]
            
#             # remove the old augmented data
#             # X= X[ np.sum(num_augmented_per_class):,:]
#             # y= y[ np.sum(num_augmented_per_class):]
            
#             #go over each row (data point), only keep the argmax prob
#             max_prob=[0]*pseudo_labels_prob.shape[0]
            
#             max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
#             for jj in range(pseudo_labels_prob.shape[0]): 
#                 idxMax=np.argmax(pseudo_labels_prob[jj,:])
                
#                 max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
#                 max_prob[jj]=pseudo_labels_prob[jj,idxMax]
        
#             # for each class, count the number of points > threshold
#             countVector=[0]*self.nClass
#             for cc in range(self.nClass):
#                 temp=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
#                 countVector[cc]= len( temp )
                
#             if np.max(countVector)==0:
#                 #print("==========",countVector)
#                 print( "len(self.unlabelled_data)",len(self.unlabelled_data), "pseudo_labels_prob.shape",pseudo_labels_prob.shape)
                
#                 return self.test_acc
            
#             countVector_normalized=np.asarray(countVector)/np.max(countVector)

#             if self.verbose:    
#                 print("class threshold:", np.round(countVector_normalized*self.upper_threshold,2))
            

#             for cc in range(self.nClass):
#                 # compute the adaptive threshold for each class
#                 class_upper_thresh=countVector_normalized[cc]*self.upper_threshold

#                 idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        

#                 temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > class_upper_thresh)[0]
#                 labels_within_threshold= idx_sorted[temp_idx]
                
#                 max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)

#                 if max_prob_matrix.shape[0]<=1:
#                     return self.test_acc
         
#                 X,y = self.post_processing(cc,labels_within_threshold,X,y)

#                 # num_augmented_per_class[cc]=len(pseudo_labels)
#                 # self.selected_unlabelled_index += labels_within_threshold.tolist()

#                 # # augmenting the data [ pseudo_data, true_data]
#                 # X = np.vstack((chosen_unlabelled_rows, X))
#                 # y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))

#             if self.verbose:    
#                 print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
#             if np.sum(self.num_augmented_per_class)==0:
#                 return self.test_acc
            
            
#             # Shuffle 
#             indices = list(range(X.shape[0]))
#             np.random.shuffle(indices)

#             X = X[indices]
#             y = y[indices]    
#         return self.test_acc

    
class flex_pl(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="flex"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self):
        return super().get_max_pseudo_point()
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        self.num_augmented_per_class=[0]*self.nClass

        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            num_points=pseudo_labels_prob.shape[0]
            # remove the old augmented data
            # X= X[ np.sum(num_augmented_per_class):,:]
            # y= y[ np.sum(num_augmented_per_class):]
            
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
            

            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                class_upper_thresh=countVector_normalized[cc]*self.upper_threshold


                MaxPseudoPoint=self.get_max_pseudo_point()
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        

                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > class_upper_thresh)[0]
                labels_within_threshold= idx_sorted[temp_idx]
                
                #labels_within_threshold = np.where(max_prob_matrix[:,cc] > class_upper_thresh)[0]
            
                # Use argmax to find the class with the highest probability
                # pseudo_labels = [cc]*len(labels_within_threshold)
                # pseudo_labels=np.asarray(pseudo_labels)
                # chosen_unlabelled_rows = self.unlabelled_data[labels_within_threshold]
    
                # num_augmented_per_class[cc]=len(pseudo_labels)
                # self.selected_unlabelled_index += labels_within_threshold.tolist()
                
                # # augmenting the data [ pseudo_data, true_data]
                # X = np.vstack((chosen_unlabelled_rows, X))
                # y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))

                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            
            # Shuffle 
            # indices = list(range(X.shape[0]))
            # np.random.shuffle(indices)

            # X = X[indices]
            # y = y[indices]    
        return self.test_acc
    

class sinkhorn_original(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="sinkhorn_original"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self):
        return super().get_max_pseudo_point()
        
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        
        self.num_augmented_per_class=[0]*self.nClass
        
        # estimate the frequency
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
    
            dist_points=np.ones( num_points+1 )
            
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=1
            C=1-pseudo_labels_prob # cost # expand Cost matrix

            C=np.vstack((C,np.ones((1,self.nClass))))
            C=np.hstack((C,np.ones((num_points+1,1))))
            
            K=np.exp(-C/regulariser)
            
            
            # expand marginal dist for columns (class)
            #dist_labels=np.ones( pseudo_labels_prob.shape[1] )*(0.5*pseudo_labels_prob.shape[0])
            dist_labels = self.label_frequency*np.sum(dist_points)-1.0/self.nClass  # frequency of the class label
            dist_labels = np.append(dist_labels,1)
            
            
            if np.abs( np.sum(dist_labels) - np.sum(dist_points) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            uu=np.ones( (num_points+1,))
            
            #uu=dist_points
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= dist_labels / np.dot(K.T, uu)
                uu= dist_points / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            #pseudo_labels_prob=temp[:-1,:-1]
            temp2 = ot.sinkhorn( dist_points , dist_labels, reg=regulariser, M=C )
            
            #pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
            pseudo_labels_prob=temp2[:-1,:-1]

            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)

            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                MaxPseudoPoint=self.get_max_pseudo_point()
                
                MaxPseudoPoint=np.int(self.FractionAllocatedLabel*num_points*self.label_frequency[cc]/self.num_iters)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
                        
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx]
            
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                        
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]    
        return self.test_acc
    
class sinkhorn_label_assignment(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="sinkhorn_label_assignment"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self):
        return super().get_max_pseudo_point()
            
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
                
        self.num_augmented_per_class=[0]*self.nClass
        
        # estimate the frequency
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
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
            vu=np.mean(pseudo_labels_prob_list,axis=0) # debug
        
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
            
            lcb_ucb=[0]*num_points
            for jj in range(num_points):# go over each row (data points)
            
                idxmax =idxargmax[jj]
                idx2ndmax=idx2nd_argmax[jj] 
                
                uncertainty_rows_argmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idxmax  ])
                uncertainty_rows_arg2ndmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idx2ndmax])
                
                # lcb of argmax - ucb of arg2ndmax
                lcb_ucb[jj] = ( pseudo_labels_prob[jj, idxmax] - kappa*uncertainty_rows_argmax[jj] ) \
                                    - (pseudo_labels_prob[jj, idx2ndmax] + kappa*uncertainty_rows_arg2ndmax[jj])
            
            
            # print("max uncertainty",np.max(uncertainty_rows_argmax),\
            #       "mean",np.mean(uncertainty_rows_argmax),"min",np.min(uncertainty_rows_argmax))
            #uncert_per_point = np.sum(uncertainty_rows,axis=1)
            #uncert_per_point=(uncert_per_point - np.min(uncert_per_point))/( np.max(uncert_per_point) - np.min(uncert_per_point))
            
            #pseudo_labels_prob=(pseudo_labels_prob-np.min(pseudo_labels_prob))/(np.max(pseudo_labels_prob)-np.min(pseudo_labels_prob))
            # perform SLA given the prob matrix
            
            # expand marginal distribution for rows (points)
            lcb_ucb=np.asarray(lcb_ucb)
            lcb_ucb=np.clip(lcb_ucb, a_min=0,a_max=np.max(lcb_ucb))
            
            
            # for numerical stability of OT, select the nonzero entry only
            idxNoneZero=np.where(lcb_ucb>0)[0]
            num_points= len(idxNoneZero)
            if len(idxNoneZero)==0: # terminate if could not find any point satisfying constraints
                return self.test_acc
            
            dist_points=lcb_ucb[idxNoneZero]
            
            
            #dist_points=np.ones( num_points ) - uncert_per_point
            
            dist_points = np.append(dist_points,1+self.nClass)
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=1
            C=1-pseudo_labels_prob # cost # expand Cost matrix
            C=C[idxNoneZero,:]

            C=np.vstack((C,np.ones((1,self.nClass))))
            C=np.hstack((C,np.ones((len(idxNoneZero)+1,1))))
            
            K=np.exp(-C/regulariser)
            
            # plt.figure(figsize=(20,5))
            # plt.imshow(K.T)
            
            
            
            # expand marginal dist for columns (class)
            #dist_labels=np.ones( pseudo_labels_prob.shape[1] )*(0.5*pseudo_labels_prob.shape[0])
            dist_labels = self.label_frequency*np.sum(dist_points)-1.0/self.nClass  # frequency of the class label
            dist_labels = np.append(dist_labels,1)
            
            
            if np.abs( np.sum(dist_labels) - np.sum(dist_points) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            #uu=np.ones( (num_points+1,))
            uu=np.ones( (len(idxNoneZero)+1,))
            
            #uu=dist_points
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= dist_labels / np.dot(K.T, uu)
                uu= dist_points / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            #pseudo_labels_prob=temp[:-1,:-1]
            temp2 = ot.sinkhorn( dist_points , dist_labels, reg=regulariser, M=C )
            
            pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
            pseudo_labels_prob[idxNoneZero,:]=temp2[:-1,:-1]

            # remove the old augmented data
            # X= X[ np.sum(num_augmented_per_class):,:]
            # y= y[ np.sum(num_augmented_per_class):]
            
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)

            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                #idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing   
                
                MaxPseudoPoint=self.get_max_pseudo_point()
                
                MaxPseudoPoint=np.int(self.FractionAllocatedLabel*num_points*self.label_frequency[cc]/self.num_iters)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
            
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx]
            
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                        
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]    
        return self.test_acc
    
class sla_knowledge_uncertainty(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="sinkhorn_label_assignment with knowledge uncertainty"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self):
        return super().get_max_pseudo_point()
            
    def entropy_prediction(self,pred):
        # pred [nPoint, nClass]
        
        ent=[0]*pred.shape[0]
        
        for ii in range(pred.shape[0]):
            ent[ii]= - np.sum( pred[ii,:]*np.log(pred[ii,:]))
            #ent[ii]=ent[ii]/pred.shape[1]
            
        return np.asarray(ent)

    def data_uncertainty(self,pred):
        # pred [nModel, nPoint, nClass]
        
        ent=np.zeros((pred.shape[0],pred.shape[1]))
        for mm in range(pred.shape[0]):
            ent[mm,:]= self.entropy_prediction(pred[mm,:,:])

        return np.mean(ent,axis=0)
    
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        self.num_augmented_per_class=[0]*self.nClass
        
        # estimate the frequency
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            for tt in range(self.num_XGB_models):
                self.model_list[tt].fit(X, y.ravel())
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list) #[nModel,nPoint,nClass]
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
            #uncertainty_rows_argmax=[0]*num_points
            #uncertainty_rows_arg2ndmax=[0]*num_points
            
            total_uncertainty=self.entropy_prediction(pseudo_labels_prob)
            
            data_uncertainty=self.data_uncertainty(pseudo_labels_prob_list)
            
            knowledge_uncertainty = total_uncertainty-data_uncertainty
            
            # scale [0,1]
            nominator=knowledge_uncertainty - np.min(knowledge_uncertainty) 
            denominator=np.max(knowledge_uncertainty) - np.min(knowledge_uncertainty)
            knowledge_uncertainty=nominator / denominator
            
            # print("max knowledge_uncertainty",np.max(knowledge_uncertainty),\
            #       "mean",np.mean(knowledge_uncertainty),"min",np.min(knowledge_uncertainty))

            # lcb_ucb=[0]*num_points
            # for jj in range(num_points):# go over each row (data points)
            
            #     idxmax =idxargmax[jj]
            #     idx2ndmax=idx2nd_argmax[jj] 
                
            #     #uncertainty_rows_argmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idxmax  ])
            #     #uncertainty_rows_arg2ndmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idx2ndmax])
                
            #     # lcb of argmax - ucb of arg2ndmax
            #     lcb_ucb[jj] = ( pseudo_labels_prob[jj, idxmax] - kappa*knowledge_uncertainty[jj] ) \
            #                         - (pseudo_labels_prob[jj, idx2ndmax] + kappa*knowledge_uncertainty[jj])
            

            # # expand marginal distribution for rows (points)
            # lcb_ucb=np.asarray(lcb_ucb)
            # lcb_ucb=np.clip(lcb_ucb, a_min=0,a_max=np.max(lcb_ucb))
            
                  
            dist_points= 1 - knowledge_uncertainty
            
            #dist_points=np.ones( num_points ) - uncert_per_point
            
            dist_points = np.append(dist_points,1+self.nClass)
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=1
            C=1-pseudo_labels_prob # cost # expand Cost matrix

            C=np.vstack((C,np.ones((1,self.nClass))))
            C=np.hstack((C,np.ones((num_points+1,1))))
            
            K=np.exp(-C/regulariser)

            # expand marginal dist for columns (class)
            #dist_labels=np.ones( pseudo_labels_prob.shape[1] )*(0.5*pseudo_labels_prob.shape[0])
            dist_labels = self.label_frequency*np.sum(dist_points)-1.0/self.nClass  # frequency of the class label
            dist_labels = np.append(dist_labels,1)
            
            
            if np.abs( np.sum(dist_labels) - np.sum(dist_points) ) > 0.001 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
            #uu=np.ones( (num_points+1,))
            uu=np.ones( (num_points+1,))
            
            #uu=dist_points
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= dist_labels / np.dot(K.T, uu)
                uu= dist_points / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp= np.atleast_2d(uu).T*(K*vv.T)
            #pseudo_labels_prob=temp[:-1,:-1]
            #temp2 = ot.sinkhorn( dist_points , dist_labels, reg=regulariser, M=C )
            
            pseudo_labels_prob=temp[:-1,:-1]

            # remove the old augmented data
            # X= X[ np.sum(num_augmented_per_class):,:]
            # y= y[ np.sum(num_augmented_per_class):]
            
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)

            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                #idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing   
                
                MaxPseudoPoint=self.get_max_pseudo_point()
                
                MaxPseudoPoint=np.int(self.FractionAllocatedLabel*num_points*self.label_frequency[cc]/self.num_iters)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
            
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx]
            
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                        
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]    
        return self.test_acc    
    

class lcb_ucb(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="lcb_ucb"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self):
        return super().get_max_pseudo_point()
    def fit(self, X, y):
        print("===================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        self.num_augmented_per_class=[0]*self.nClass

        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        self.label_frequency=label_frequency/np.sum(label_frequency)
        
        
        
        # estimate the frequency
        # unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        # label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
        # self.label_frequency=label_frequency/np.sum(label_frequency)
        # #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        # print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
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
            vu=np.mean(pseudo_labels_prob_list,axis=0) # debug
        
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
            
          
            # plt.figure(figsize=(20,4))
            # plt.imshow(lcb_ucb)

            # remove the old augmented data
            # X= X[ np.sum(num_augmented_per_class):,:]
            # y= y[ np.sum(num_augmented_per_class):]
            
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(lcb_ucb)

            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
            
                MaxPseudoPoint=self.get_max_pseudo_point()

                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
            
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
                labels_within_threshold= idx_sorted[temp_idx]
            
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
                        
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]    
        return self.test_acc
    
    
    
# class sinkhorn_label_assignment_2(pseudo_labeling_iterative):#without extra dimension
    
#     def __init__(self, model, unlabelled_data, x_test,y_test,
#                  upper_threshold = 0.9, lower_threshold = 0.1, num_iters=10,flagRepetition=False,verbose = False):
#         super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
#         self.name="sinkhorn_label_assignment 2"
        
#     def predict(self, X):
#         super().predict(X)
#     def predict_proba(self, X):
#         super().predict_proba(X)
#     def evaluate(self):
#         super().evaluate()
#     def get_max_pseudo_point(self):
#         return super().get_max_pseudo_point()
#     def fit(self, X, y):
#         print("===================",self.name)
#         self.nClass=len(np.unique(y))
#         if len(np.unique(y)) < len(np.unique(self.y_test)):
#             print("num class in training data is less than test data !!!")
            
#         self.num_augmented_per_class=[0]*self.nClass
        
#         # estimate the frequency
#         unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        
#         label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

#         self.label_frequency=label_frequency/np.sum(label_frequency)
#         #label_frequency=np.ones( self.nClass )-label_frequency
#         #label_frequency=np.ones( self.nClass )*1.0/self.nClass
#         print("==label_frequency", np.round(self.label_frequency,3))


#         for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            
            
#             # Fit to data
#             self.model.fit(X, y.ravel())
            
#             self.evaluate()
            
#             num_points=self.unlabelled_data.shape[0]

#             # estimate prob using unlabelled data
#             #pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            
#             for tt in range(self.num_XGB_models):
#                 self.model_list[tt].fit(X, y.ravel())
                                                
#             # estimate prob using unlabelled data
#             pseudo_labels_prob_list=[0]*self.num_XGB_models
#             for tt in range(self.num_XGB_models):
#                 pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
#             pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
#             pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
#             vu= np.mean(pseudo_labels_prob_list,axis=0)

#             # get argmax in each row and second best 2nd argmax in each row
#             #idxargmax = np.argmax(pseudo_labels_prob_list, axis=0)
#             temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
#             idxargmax=temp[:,0]
#             idx2nd_argmax= temp[:,1]
#             # get uncertainty estimation in argmax and 2nd argmax
            
#             kappa=1
                        
#             # calculate uncertainty estimation for each data points===============
#             #uncertainty_rows=np.ones((pseudo_labels_prob.shape))
#             uncertainty_rows_argmax=[0]*num_points
#             uncertainty_rows_arg2ndmax=[0]*num_points
            
#             lcb_ucb=[0]*num_points
#             for jj in range(num_points):# go over each row (data points)
            
#                 idxmax =idxargmax[jj]
#                 idx2ndmax=idx2nd_argmax[jj] 
                
#                 uncertainty_rows_argmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idxmax  ])
#                 uncertainty_rows_arg2ndmax[jj]=np.std(pseudo_labels_prob_list[:,jj,idx2ndmax])
                
#                 # lcb of argmax - ucb of arg2ndmax
#                 lcb_ucb[jj] = ( pseudo_labels_prob[jj, idxmax] - kappa*uncertainty_rows_argmax[jj] ) \
#                                     - (pseudo_labels_prob[jj, idx2ndmax] + kappa*uncertainty_rows_arg2ndmax[jj])
            
#             #uncert_per_point = np.sum(uncertainty_rows,axis=1)
#             #uncert_per_point=(uncert_per_point - np.min(uncert_per_point))/( np.max(uncert_per_point) - np.min(uncert_per_point))
            
#             #pseudo_labels_prob=(pseudo_labels_prob-np.min(pseudo_labels_prob))/(np.max(pseudo_labels_prob)-np.min(pseudo_labels_prob))
#             # perform SLA given the prob matrix
#             regulariser=1
#             C=1-pseudo_labels_prob # cost # expand Cost matrix
#             #C=np.vstack((C,np.ones((1,self.nClass))))
#             #C=np.hstack((C,np.ones((num_points+1,1))))
            
#             K=np.exp(-C/regulariser)
            
#             # expand marginal distribution for rows (points)
#             lcb_ucb=np.asarray(lcb_ucb)
#             lcb_ucb=np.clip(lcb_ucb, a_min=0,a_max=np.max(lcb_ucb))
            
#             idxNoneZero=np.where(lcb_ucb>0)[0]
            
#             if len(idxNoneZero)==0: # terminate if could not find any point satisfying constraints
#                 return self.test_acc

            
#             K=K[idxNoneZero,:]
#             C=C[idxNoneZero,:]
            
#             dist_points=lcb_ucb[idxNoneZero]
#             #dist_points=np.ones( num_points ) - uncert_per_point

#             #dist_points = np.append(dist_points,1+self.nClass)
#             #dist_points=dist_points/np.max(dist_points)            
            
#             # expand marginal dist for columns (class)
#             #dist_labels=np.ones( pseudo_labels_prob.shape[1] )*(0.5*pseudo_labels_prob.shape[0])
#             dist_labels = self.label_frequency*np.sum(dist_points)#-1.0/self.nClass  # frequency of the class label
            
#             if np.abs( np.sum(dist_labels) - np.sum(dist_points) ) > 0.001 :
#                 print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")
            
#             #dist_labels = np.append(dist_labels,1)
#             #uu=np.ones( (num_points+1,))
#             #uu=np.ones( (num_points,))
#             uu=np.ones( (len(idxNoneZero),))

#             #uu=dist_points
#             #vv=np.ones( (pseudo_labels_prob.shape[1],))
#             for jj in range(100):
#                 vv= dist_labels / np.dot(K.T, uu)
#                 uu= dist_points / np.dot(K, vv)
                
            
#             # recompute pseudo-label-prob from SLA
#             temp= np.atleast_2d(uu).T*(K*vv.T)
#             #pseudo_labels_prob=temp
#             temp2 = ot.sinkhorn( dist_points , dist_labels, reg=regulariser, M=C )
#             #pseudo_labels_prob=temp2
            
#             pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
#             pseudo_labels_prob[idxNoneZero,:]=temp2

#             # remove the old augmented data
#             # X= X[ np.sum(num_augmented_per_class):,:]
#             # y= y[ np.sum(num_augmented_per_class):]
            
#             #go over each row (data point), only keep the argmax prob            
#             max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)

#             for cc in range(self.nClass):
#                 # compute the adaptive threshold for each class
            
#                 MaxPseudoPoint=self.get_max_pseudo_point()

#                 idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
            
#                 temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > 0 )[0] #threshold 0.5
#                 labels_within_threshold= idx_sorted[temp_idx]
            
#                 max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
#                 if max_prob_matrix.shape[0]<=1:
#                     return self.test_acc
            
#                 X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
#             if self.verbose:
#                 print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
#             if np.sum(self.num_augmented_per_class)==0:
#                 return self.test_acc
                        
#             # Shuffle 
#             indices = list(range(X.shape[0]))
#             np.random.shuffle(indices)

#             X = X[indices]
#             y = y[indices]    
#         return self.test_acc
    
# entropy estimation
class entropy_pl(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.8, lower_threshold = 0.2, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="entropy_pl"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy= - np.sum(vector_prob.*np.log(vector_prob))
        entropy_score = entropy(vector_prob,base=2)
        return entropy_score
    
    def calculate_gap_best_2nd_best(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        gap=np.max(vector_prob)-second_best[-2]
        return gap
    

    def fit(self, X, y):
        print("====================",self.name)
        
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        self.num_augmented_per_class=[0]*self.nClass


        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            self.evaluate()
         
                
            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            # remove the old augmented data
            #X= X[ np.sum(self.num_augmented_per_class):,:]
            #y= y[ np.sum(self.num_augmented_per_class):]
        
            entropy_rows=2*np.zeros((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                entropy_rows[jj,idxMax]=self.calculate_gap_best_2nd_best(pseudo_labels_prob[jj,:])
        
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                class_upper_thresh=self.upper_threshold

                idx_sorted = np.argsort( entropy_rows[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        

                #idx_sorted = np.argsort( entropy_rows[:,cc])[:self.MaxPseudoPoint_per_Class] # increasing

                temp_idx = np.where(entropy_rows[idx_sorted,cc] > class_upper_thresh)[0]
                labels_within_threshold= idx_sorted[temp_idx]
            
                #print("cc ",cc, "#augmented ", len(labels_within_threshold))
                
                entropy_rows = np.delete(entropy_rows, labels_within_threshold, 0)
                if entropy_rows.shape[0]<=1:
                    return self.test_acc
                
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc

# entropy estimation and prediction score
class prediction_entropy(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.8, lower_threshold = 0.2, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="prediction_entropy"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_prob_at_max_class(self,pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy_score = entropy(vector_prob,base=2)
        
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        entropy_score=np.max(vector_prob)-second_best[-2]
        return entropy_score

    def fit(self, X, y):
        
        print("====================",self.name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        self.num_augmented_per_class=[0]*self.nClass

        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            self.evaluate()
         
                
            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            # remove the old augmented data
            #X= X[ np.sum(num_augmented_per_class):,:]
            #y= y[ np.sum(num_augmented_per_class):]
        
            entropy_rows=np.zeros((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                entropy_rows[jj,idxMax]=self.calculate_entropy(pseudo_labels_prob[jj,:])
        
            #go over each row (data point), only keep the argmax prob
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
            
            # larger is betetr
            #acq_func=(1-entropy_rows)+max_prob_matrix
            #acq_func=0.5*entropy_rows + 0.5*max_prob_matrix
            
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                class_upper_thresh=self.upper_threshold

                # idx_sorted = np.argsort( acq_func[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        
                # temp_idx = np.where(acq_func[idx_sorted,cc] > class_upper_thresh)[0]
                # labels_within_threshold= idx_sorted[temp_idx]
                
                
                temp_idx1 = np.where( max_prob_matrix[:,cc] > self.upper_threshold +0.1)[0]
                temp_idx2 = np.where( entropy_rows[:,cc] < self.upper_threshold -0.1)[0]
                labels_within_threshold =np.intersect1d(temp_idx1, temp_idx2)
               
                
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                entropy_rows = np.delete(entropy_rows, labels_within_threshold, 0)
            
            
                #acq_func = np.delete(acq_func, labels_within_threshold, 0)
                if entropy_rows.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc
    
    
    
# uncertainty estimation

#1. generate multiple XGB with different hyper

# uncertainty estimation and prediction score
class uncertainty_prediction(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.8, lower_threshold = 0.2, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="uncertainty_prediction"

      

    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def cross_entropy_matrix(self,probA, probB):
        return super().cross_entropy_matrix(probA, probB)
    def uncertainty_score(self,matrixA):
        return super().uncertainty_score(matrixA)

        
    def get_prob_at_max_class(self,pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy_score = entropy(vector_prob,base=2)
        
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        entropy_score=np.max(vector_prob)-second_best[-2]
        return entropy_score

    def fit(self, X, y):
        print("====================",self.name)

        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")

        self.num_augmented_per_class=[0]*self.nClass

        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            self.evaluate()

            for tt in range(self.num_XGB_models):
                self.model_list[tt].fit(X, y.ravel())
                
                #y_test_pred = self.model_list[tt].predict(self.x_test)
                #test_acc[tt]=accuracy_score(y_test_pred, self.y_test)*100
                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                pseudo_labels_prob_list[tt] = self.model_list[tt].predict_proba(self.unlabelled_data)
        
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
        
            # calculate uncertainty estimation for each data points
            uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                uncertainty_rows[jj,idxMax]=np.std(pseudo_labels_prob_list[:,jj,idxMax])
                #uncertainty_rows[jj,idxMax]=self.uncertainty_score(pseudo_labels_prob_list[:,jj,:])
            
            
            # entropy_rows=np.zeros((pseudo_labels_prob.shape))
            # for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
            #     idxMax=np.argmax( pseudo_labels_prob[jj,:] )
            #     entropy_rows[jj,idxMax]=self.calculate_entropy(pseudo_labels_prob[jj,:])
        
        
        
            #go over each row (data point), only keep the argmax prob
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
            
            # larger is betetr
            #acq_func=(1-entropy_rows)+max_prob_matrix
            #acq_func=0.6*entropy_rows + 0.5*max_prob_matrix
            #acq_func=0.5*(max_prob_matrix/np.max(max_prob_matrix)) + 0.5*(1- uncertainty_rows/np.max(uncertainty_rows))
            #acq_func=0.5*(max_prob_matrix) + 0.5*(1- uncertainty_rows)
            
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                #class_upper_thresh=self.upper_threshold

                # idx_sorted = np.argsort( acq_func[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        
                # temp_idx = np.where(acq_func[idx_sorted,cc] > class_upper_thresh)[0]
                # labels_within_threshold= idx_sorted[temp_idx]


                temp_idx1 = np.where( max_prob_matrix[:,cc] > self.upper_threshold +0.1)[0]
                temp_idx2 = np.where( uncertainty_rows[:,cc] < self.lower_threshold)[0]
                labels_within_threshold =np.intersect1d(temp_idx1, temp_idx2)
               
                
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                uncertainty_rows = np.delete(uncertainty_rows, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc
    
# uncertainty estimation and prediction score
class uncertainty_pl(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.8, lower_threshold = 0.2, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
        self.name="uncertainty"

        
        # generate multiple models
        # params = { 'max_depth': np.arange(3, 20, 18).astype(int),
        #            'learning_rate': [0.01, 0.1, 0.2, 0.3],
        #            'subsample': np.arange(0.5, 1.0, 0.05),
        #            'colsample_bytree': np.arange(0.4, 1.0, 0.05),
        #            'colsample_bylevel': np.arange(0.4, 1.0, 0.05),
        #            'n_estimators': [100, 200, 300, 500, 600, 700, 1000]}
        
        # self.model_list=[0]*self.num_XGB_models
        
        # param_list=[0]*self.num_XGB_models
        # for tt in range(self.num_XGB_models):
            
        #     param_list[tt]={}
            
        #     for key in params.keys():
        #         mychoice=random.choice(params[key])
            
        #         param_list[tt][key]=mychoice
        #         param_list[tt]['verbosity'] = 0
        #         param_list[tt]['silent'] = 1
                
        #     self.model_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)


    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        return super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def cross_entropy_matrix(self,probA, probB):
        return super().cross_entropy_matrix(probA, probB)
    def uncertainty_score(self, matrix_prob):
        return super().uncertainty_score(matrix_prob)

        
    def get_prob_at_max_class(self,pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy_score = entropy(vector_prob,base=2)
        
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        entropy_score=np.max(vector_prob)-second_best[-2]
        return entropy_score

    def fit(self, X, y):
        print("====================",self.name)

        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")

        self.num_augmented_per_class=[0]*self.nClass

        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

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
        
            # calculate uncertainty estimation for each data points
            uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                uncertainty_rows[jj,idxMax]=np.std(pseudo_labels_prob_list[:,jj,idxMax])
                #uncertainty_rows[jj,idxMax]=self.uncertainty_score(pseudo_labels_prob_list[:,jj,idxMax])
            
            
            # entropy_rows=np.zeros((pseudo_labels_prob.shape))
            # for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
            #     idxMax=np.argmax( pseudo_labels_prob[jj,:] )
            #     entropy_rows[jj,idxMax]=self.calculate_entropy(pseudo_labels_prob[jj,:])
        
        
        
            #go over each row (data point), only keep the argmax prob
            #max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
            
            # larger is betetr
            #acq_func=(1-entropy_rows)+max_prob_matrix
            #acq_func=0.6*entropy_rows + 0.5*max_prob_matrix
            #acq_func=0.5*(max_prob_matrix/np.max(max_prob_matrix)) + 0.5*(1- uncertainty_rows/np.max(uncertainty_rows))
            acq_func=1- uncertainty_rows
            
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                class_upper_thresh=self.upper_threshold

                idx_sorted = np.argsort( acq_func[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        

                temp_idx = np.where(acq_func[idx_sorted,cc] > class_upper_thresh)[0]
                labels_within_threshold= idx_sorted[temp_idx]

                acq_func = np.delete(acq_func, labels_within_threshold, 0)
                if acq_func.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc
            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc
    
    
# uncertainty estimation and prediction score
class uncertainty_entropy_score(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, \
                 num_iters=10,flagRepetition=False,verbose = False, datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,\
                         num_iters,flagRepetition,verbose,datasetName)
        
        self.name="uncertainty entropy prediction"

      
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
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy_score = entropy(vector_prob,base=2)
        
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        entropy_score=np.max(vector_prob)-second_best[-2]
        return entropy_score
    
    def calculate_gap_best_2nd_best(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        gap=np.max(vector_prob)-second_best[-2]
        return gap

    def fit(self, X, y):
        
        print("====================",self.name)

        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
                    
        self.num_augmented_per_class=[0]*self.nClass

        for ii in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

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
            
            countVector=[0]*self.nClass
            for cc in range(self.nClass):
                temp=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
                countVector[cc]= len( temp )
                
            if np.max(countVector)==0:
                #print("==========",countVector)
                print( "len(self.unlabelled_data)",len(self.unlabelled_data), "pseudo_labels_prob.shape",pseudo_labels_prob.shape)
                
                return self.test_acc
            
            # calculate uncertainty estimation for each data points
            uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                uncertainty_rows[jj,idxMax]=np.std(pseudo_labels_prob_list[:,jj,idxMax])
                #uncertainty_rows[jj,idxMax]=self.uncertainty_score(pseudo_labels_prob_list[:,jj,:])
            
            
            entropy_rows=np.zeros((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                entropy_rows[jj,idxMax]=self.calculate_gap_best_2nd_best(pseudo_labels_prob[jj,:])
        
            # if self.verbose:

            #     print("entropy_rows",entropy_rows[:10,:])
            #     print("uncertainty_rows",uncertainty_rows[:10,:])
            #     print("max_prob_matrix",max_prob_matrix[:10,:])

            # plot
            IsPlot=False
            if IsPlot:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                            
                # For each set of style and range settings, plot n random points in the box
                # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
                acq_func=0.33*np.max(max_prob_matrix,axis=1) + 0.33*np.max(entropy_rows,axis=1) \
                    + 0.34*(1- np.min(uncertainty_rows,axis=1))
                    
                marker_list=['*','o','v','s','^','x','>','<','D','+','1','2']
                
                for jj in range(self.nClass):
                    idx=np.where( np.argmax(max_prob_matrix,axis=1) ==jj)[0]
                    
                    acq_func=0.33*np.max(max_prob_matrix[idx,:],axis=1) + 0.33*np.max(entropy_rows[idx,:],axis=1) \
                        + 0.34*(1- np.min(uncertainty_rows[idx,:],axis=1))
                        
                #marker_toplot= [marker_list[idx] for idx in np.argmax(max_prob_matrix,axis=1)]
                    p=ax.scatter( np.min(uncertainty_rows[idx,:],axis=1), np.max(entropy_rows[idx,:],axis=1)\
                               ,np.max(max_prob_matrix[idx,:],axis=1) , c=acq_func  , \
                                   marker=marker_list[jj])
                    
                ax.set_xlabel('Uncertainty')
                ax.set_ylabel('Entropy')
                ax.set_zlabel('Prob')
                
                fig.colorbar(p)
    
                strOut="{:s}_{:d}_3d.pdf".format(self.datasetName,ii)
                fig.savefig(strOut)

            
            # larger is betetr
            #acq_func=(1-entropy_rows)+max_prob_matrix
            #acq_func=0.6*entropy_rows + 0.5*max_prob_matrix

            
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                #class_upper_thresh=self.upper_threshold

                idx_sorted = np.argsort( entropy_rows[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        
                temp_idx1 = np.where(entropy_rows[idx_sorted,cc] > self.upper_threshold)[0]
                # labels_within_threshold= idx_sorted[temp_idx]
            
                #temp_idx1 = np.where( entropy_rows[:,cc] > self.upper_threshold -0.1)[0]
                temp_idx2 = np.where( max_prob_matrix[:,cc] > self.upper_threshold +0.1)[0]
                temp_idx3 = np.where( uncertainty_rows[:,cc] < self.lower_threshold)[0]
                labels_within_threshold =np.intersect1d(temp_idx1, temp_idx2)
                labels_within_threshold =np.intersect1d(labels_within_threshold, temp_idx3)
               
                
                entropy_rows = np.delete(entropy_rows, labels_within_threshold, 0)
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                uncertainty_rows = np.delete(uncertainty_rows, labels_within_threshold, 0)

                if entropy_rows.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc

            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc
    
    
class uncertainty_entropy(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,
                 upper_threshold = 0.8, lower_threshold = 0.2, num_iters=10,flagRepetition=False,verbose = False):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,num_iters,flagRepetition,verbose)
  

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
    def calculate_entropy(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        #entropy_score = entropy(vector_prob,base=2)
        
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        entropy_score=np.max(vector_prob)-second_best[-2]
        return entropy_score
    
    def calculate_gap_best_2nd_best(self,vector_prob):
        vector_prob=vector_prob/np.sum(vector_prob)
        # gap between the best and second best
        second_best = np.sort(vector_prob) # increasing
        gap=np.max(vector_prob)-second_best[-2]
        return gap

    def fit(self, X, y):
        print("====================",self.name)

        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        
        self.num_augmented_per_class=[0]*self.nClass

        for _ in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

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
            
            countVector=[0]*self.nClass
            for cc in range(self.nClass):
                temp=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
                countVector[cc]= len( temp )
                
            if np.max(countVector)==0:
                #print("==========",countVector)
                print( "len(self.unlabelled_data)",len(self.unlabelled_data), "pseudo_labels_prob.shape",pseudo_labels_prob.shape)
                
                return self.test_acc
        
            # calculate uncertainty estimation for each data points
            uncertainty_rows=np.ones((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                uncertainty_rows[jj,idxMax]=np.std(pseudo_labels_prob_list[:,jj,idxMax])
                #uncertainty_rows[jj,idxMax]=self.uncertainty_score(pseudo_labels_prob_list[:,jj,:])
            
            entropy_rows=np.zeros((pseudo_labels_prob.shape))
            for jj in range(pseudo_labels_prob.shape[0]):# go over each row (data points)
                idxMax=np.argmax( pseudo_labels_prob[jj,:] )
                entropy_rows[jj,idxMax]=self.calculate_gap_best_2nd_best(pseudo_labels_prob[jj,:])
        
            # larger is betetr
            #acq_func=(1-entropy_rows)+max_prob_matrix
            #acq_func=0.6*entropy_rows + 0.5*max_prob_matrix
            #acq_func=0.5*(entropy_rows/np.max(entropy_rows)) + 0.5*(1- uncertainty_rows/np.max(uncertainty_rows))
            #acq_func=0.5*(max_prob_matrix) + 0.5*(1- uncertainty_rows)
            
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                #class_upper_thresh=countVector_normalized[cc]*self.upper_threshold
                #class_upper_thresh=self.upper_threshold

                #idx_sorted = np.argsort( acq_func[:,cc])[::-1][:self.MaxPseudoPoint_per_Class] # decreasing        
                #temp_idx = np.where(acq_func[idx_sorted,cc] > class_upper_thresh)[0]
                #labels_within_threshold= idx_sorted[temp_idx]
                
                temp_idx1 = np.where( entropy_rows[:,cc] > self.upper_threshold -0.1)[0]
                temp_idx2 = np.where( uncertainty_rows[:,cc] < self.lower_threshold)[0]
                labels_within_threshold =np.intersect1d(temp_idx1, temp_idx2)
               
            
                max_prob_matrix = np.delete(max_prob_matrix, labels_within_threshold, 0)
                uncertainty_rows = np.delete(uncertainty_rows, labels_within_threshold, 0)
                if max_prob_matrix.shape[0]<=1:
                    return self.test_acc
         
                X,y = self.post_processing(cc,labels_within_threshold,X,y)

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

            if np.sum(self.num_augmented_per_class)==0:
                return self.test_acc

            
            # Shuffle 
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]   
            
        return self.test_acc