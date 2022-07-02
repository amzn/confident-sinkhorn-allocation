
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import stats
import time					
import copy

    
class Pseudo_Labeling(object):
    # implementation of the master class for pseudo-labeling
    # this class will be inherited across other subclasses
    
    def __init__(self, unlabelled_data, x_test,y_test, num_iters=5,upper_threshold = 0.8, \
                 lower_threshold = None,num_XGB_models=0,verbose = False):
        
        self.algorithm_name="Pseudo_Labeling"
        self.x_test=x_test
        self.y_test=y_test
        
        # for house keeping and reporting purpose
        self.len_unlabels=[]
        self.len_accepted_ttest=[]
        self.len_selected=[]
        #self.datasetName=datasetName 



        # this is the XGBoost model for classification
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'
        param['verbosity'] = 0
        param['silent'] = 1
        param['seed'] = 0

        # create XGBoost instance with default hyper-parameters
        xgb = XGBClassifier(**param,use_label_encoder=False)

        self.model = copy.copy(xgb)

        self.unlabelled_data = unlabelled_data # this is a temporary unlabelled data changing in each iteration
        self.verbose = verbose
        self.upper_threshold = upper_threshold
        self.num_iters=num_iters

        if lower_threshold is not None:
            self.lower_threshold = lower_threshold # this lower threshold is used for UPS algorithm, not the vanilla Pseudo-labeling
        

        # allow the pseudo-data is repeated, e.g., without removing them after each iteration        
        # create a list of all the indices 
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))      
        
        self.selected_unlabelled_index=[]

        if self.verbose:
            print("no of unlabelled data:",unlabelled_data.shape[0], "\t no of test data:",x_test.shape[0])

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)
        self.test_acc=[]
        self.FractionAllocatedLabel=1
        self.num_XGB_models=num_XGB_models # this is the parameter M in our paper

        if num_XGB_models>1: # will be used for CSA and UPS
            # for uncertainty estimation
            # generate multiple models
            params = { 'max_depth': np.arange(3, 20).astype(int),
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'subsample': np.arange(0.5, 1.0, 0.05),
                    'colsample_bytree': np.arange(0.4, 1.0, 0.05),
                    'colsample_bylevel': np.arange(0.4, 1.0, 0.05),
                    'n_estimators': [100, 200, 300, 500, 600, 700, 1000]}
            
            self.XGBmodels_list=[0]*self.num_XGB_models
            
            param_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                
                param_list[tt]={}
                
                for key in params.keys():
              
                    mychoice=np.random.choice(params[key])
                
                    param_list[tt][key]=mychoice
                    param_list[tt]['verbosity'] = 0
                    param_list[tt]['silent'] = 1
                    param_list[tt]['seed'] = tt
                    
                self.XGBmodels_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)
    
    def estimate_label_frequency(self, y):
        # estimate the label frequency empirically from the initial labeled data
        unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)

        if self.verbose:
            print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        # smooth the label frequency if the ratio between the max class / min class is significant >5
        ratio=np.max(label_frequency)/np.min(label_frequency)
        if ratio>5:
            label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
    
        return label_frequency/np.sum(label_frequency)


    
    def label_assignment_and_post_processing(self, assignment_matrix_Q,X,y, current_iter=0):
        
        num_points=assignment_matrix_Q.shape[0]
        max_prob_matrix = self.get_prob_at_max_class(assignment_matrix_Q)

        augmented_idx=[]
        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass):
            # compute the adaptive threshold for each class
            
            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        
        
            # we only accept index where max prob >0
            idx_nonzero = np.where( max_prob_matrix[idx_sorted,cc] > 0 )[0]
            
            labels_within_threshold=idx_sorted[idx_nonzero][:MaxPseudoPoint[cc]]
            
            
            augmented_idx += labels_within_threshold.tolist()

            X,y = self.post_processing(cc,labels_within_threshold,X,y)
            
        if self.verbose:
            print("MaxPseudoPoint",MaxPseudoPoint)

        if np.sum(self.num_augmented_per_class)==0:
            return X,y
        
        self.len_unlabels.append( len(self.unlabelled_data) )
        self.len_accepted_ttest.append( num_points ) 
        self.len_selected.append(  np.sum(self.num_augmented_per_class) )
        
    
    
        # remove the selected data from unlabelled data
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
        
    def evaluate(self):
        y_test_pred = self.model.predict(self.x_test)
        test_acc= np.round( accuracy_score(y_test_pred, self.y_test)*100, 2)# round to 2 digits xx.yy %

        print('+++Test Acc: {:.2f}%'.format(test_acc))
        self.test_acc +=[test_acc]
    
    def get_prob_at_max_class(self,pseudo_labels_prob):
        max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
        for ii in range(pseudo_labels_prob.shape[0]):  # loop over each data point
            idxMax=np.argmax(pseudo_labels_prob[ii,:]) # find the highest score class
            max_prob_matrix[ii,idxMax]=pseudo_labels_prob[ii,idxMax]
        return max_prob_matrix
    
    def post_processing(self,cc,labels_within_threshold,X,y):      
        # Merge the pseudo-labeled data into the existing labeled data ================

        # the data samples X
        chosen_unlabelled_rows = self.unlabelled_data[labels_within_threshold,:]
                
        # we define the pseudo-labels Y
        pseudo_labels = [cc]*len(labels_within_threshold)
        pseudo_labels=np.asarray(pseudo_labels)

        X = np.vstack((chosen_unlabelled_rows, X))
        y = np.vstack((pseudo_labels.reshape(-1,1), np.array(y).reshape(-1,1)))        

        # store the index of the unlabeled points which have been selected to assign labels
        # we later will delete these points from the unlabeled set
        self.selected_unlabelled_index += labels_within_threshold.tolist()
        
        # store the number of augmented points per class
        self.num_augmented_per_class[cc]=len(pseudo_labels)
        
        return X,y

    def get_max_pseudo_point(self,fraction_of_class, current_iter): # more at the begining and less at later stage
        
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
        print("=====",self.algorithm_name)


        self.nClass=len(np.unique(y))
        
        
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")

        self.num_augmented_per_class=[0]*self.nClass

        self.label_frequency=self.estimate_label_frequency(y)

        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y.ravel())

            # Evaluate the performance on test set after Fit the model given the data
            self.evaluate()
            
            # Predictive probability on the unlabeled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)

            # number of unlabeled data
            num_points = pseudo_labels_prob.shape[0]

            #go over each row (data point), only keep the argmax prob
            max_prob=[0]*num_points
            max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            for ii in range(num_points): 
                idxMax=np.argmax(pseudo_labels_prob[ii,:])
                
                max_prob_matrix[ii,idxMax]=pseudo_labels_prob[ii,idxMax]
                max_prob[ii]=pseudo_labels_prob[ii,idxMax]
        
            augmented_idx=[]
            for cc in range(self.nClass):

                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        
    
                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > self.upper_threshold)[0]
                labels_within_threshold= idx_sorted[temp_idx]
            
                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                augmented_idx += labels_within_threshold.tolist()

                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
         
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
                
            # remove the selected data from unlabelled data
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)   

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