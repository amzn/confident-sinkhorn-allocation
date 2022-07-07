
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
import copy
import sklearn

    
class Pseudo_Labeling(object):
    # implementation of the master class for pseudo-labeling
    # this class will be inherited across other subclasses
    
    def __init__(self, unlabelled_data, x_test,y_test, num_iters=5,upper_threshold = 0.8, \
            fraction_allocation=1,lower_threshold = None,num_XGB_models=0, \
                 verbose = False,IsMultiLabel=False):
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

        self.IsMultiLabel=False
        self.algorithm_name="Pseudo_Labeling"
        self.x_test=x_test
        self.y_test=y_test

        self.IsMultiLabel=IsMultiLabel

        # for house keeping and reporting purpose
        self.len_unlabels=[]
        self.len_accepted_ttest=[]
        self.len_selected=[]
        self.num_augmented_per_class=[]


        # this is the XGBoost model for multi-class classification
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'
        param['verbosity'] = 0
        param['silent'] = 1
        param['seed'] = 0

        # create XGBoost instance with default hyper-parameters
        #xgb = XGBClassifier(**param,use_label_encoder=False)
        xgb = self.get_XGB_model(param)

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
        self.FractionAllocatedLabel=fraction_allocation # we will allocate labels to 100% of the unlabeled dataset
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
                    
                #self.XGBmodels_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)
                self.XGBmodels_list[tt] = self.get_XGB_model(param_list[tt])


    def get_XGB_model(self,param): 
        """
        we create the XGB model depending on multiclass or multi-label setting
        Args:
            param: a predefined hyperparameter for XGBmodel

        Output:
            a single XGBClassifier for multiclass
            or
            a single MultiOutputClassifier for multilabel
        """

        if self.IsMultiLabel==False:
            return XGBClassifier(**param,use_label_encoder=False)
        else:
            return MultiOutputClassifier(XGBClassifier(**param,use_label_encoder=False))

    def get_predictive_prob_for_unlabelled_data(self, model):
        """
        Compute the predictive probability within [0,1] for unlabelled data given a single XGB model
        Args:
            model: a single XGBmodel

        Output:
            predictive probability matrix [N x K]
        """

        pseudo_labels_prob = model.predict_proba(self.unlabelled_data)
        
        # number of unlabeled data
        if self.IsMultiLabel==True:
            pseudo_labels_prob=np.asarray(pseudo_labels_prob).T
            pseudo_labels_prob=pseudo_labels_prob[1,:,:]

        return pseudo_labels_prob

    def estimate_label_frequency(self, y):
        """
        estimate the label frequency empirically from the initial labeled data
        Args:
            y: label vector or matrix (multilabel)

        Output:
            Given K the number of labels, it returns a vector of label frequency [1 x K]
        """
        

        if self.IsMultiLabel==False:
            if len(self.num_augmented_per_class)>0:
                unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
            else:
                unique, label_frequency = np.unique( y, return_counts=True)
        else:
            label_frequency = np.sum( y, axis=0)

        if self.verbose:
            print("==label_frequency without adjustment", np.round(label_frequency,3))
        
        # smooth the label frequency if the ratio between the max class / min class is significant >5
        # this smoothing is the implementation trick to prevent biased estimation given limited training data
        ratio=np.max(label_frequency)/np.min(label_frequency)
        if ratio>5:
            label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass
    
        return label_frequency/np.sum(label_frequency)

    
        
    def evaluate_performance(self):
        """
        evaluate_performance the classification performance
        Store the result into: self.test_acc which is the accuracy for multiclassification \
                                                    or the precision for multilabel classification
        """
        

        y_test_pred = self.model.predict(self.x_test)

        if self.IsMultiLabel==False:
            test_acc= np.round( accuracy_score(y_test_pred, self.y_test)*100, 2)# round to 2 digits xx.yy %

            if self.verbose:
                print('+++Test Acc: {:.2f}%'.format(test_acc))
            self.test_acc +=[test_acc]
        else: # multi-label classification

            # Precision
            prec=sklearn.metrics.precision_score(self.y_test, y_test_pred,average='samples')*100
            prec=np.round(prec,2) # round to 2 digits xx.yy %

            self.test_acc +=[prec] # precision score

            if self.verbose:
                print('+++Test Acc: {:.2f}%'.format(prec))


    def get_prob_at_max_class(self,pseudo_labels_prob):
        """
        Given the 2d probability matrix [N x K], we get the probability at the maximum index
        Args:
           pseudo_labels_prob: 2d probability matrix [N x K]

        Returns:
           max_prob_matrix: probability at argmax class [N x 1]
        """
        max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
        for ii in range(pseudo_labels_prob.shape[0]):  # loop over each data point
            idxMax=np.argmax(pseudo_labels_prob[ii,:]) # find the highest score class
            max_prob_matrix[ii,idxMax]=pseudo_labels_prob[ii,idxMax]
        return max_prob_matrix
    
    def post_processing_and_augmentation(self,assigned_pseudo_labels,X,y):
        """
        after assigning the pseudo labels in the previous step, we post-process and augment them into X and y
        Args:
            assigned_pseudo_labels: [N x K] matrix where N is the #unlabels and K is the #class
            assigned_pseudo_labels==0 indicates no assignment
            assigned_pseudo_labels==1 indicates assignment.

            X: existing pseudo_labeled + labeled data [ N' x d ]
            y: existing pseudo_labeled + labeled data [ N' x 1 ] for multiclassification
            y: existing pseudo_labeled + labeled data [ N' x K ] for multilabel classification
        Output:
            Augmented X
            Augmented y
        """

        sum_by_cols=np.sum(assigned_pseudo_labels,axis=1)            
        labels_satisfied_threshold = np.where(sum_by_cols>0)[0]
        
        self.num_augmented_per_class.append( np.sum(assigned_pseudo_labels,axis=0).astype(int) )
        
        if len(labels_satisfied_threshold) == 0: # no point is selected
            return X,y
            
        self.selected_unlabelled_index += labels_satisfied_threshold.tolist()

        # augment the assigned labels to X and y ==============================================
        X = np.vstack((self.unlabelled_data[labels_satisfied_threshold,:], X))

        if self.IsMultiLabel==False: # y is [N x 1] matrix
            # allow a single data point can be added into multiple 
            y = np.vstack(( np.argmax( assigned_pseudo_labels[labels_satisfied_threshold,:],axis=1).reshape(-1,1), np.array(y).reshape(-1,1)))  
          
        else: # y is [N x L] matrix
            y = np.vstack((assigned_pseudo_labels[labels_satisfied_threshold,:], np.array(y)))


        if "CSA" in self.algorithm_name: # book keeping
            self.len_unlabels.append( len(self.unlabelled_data) )
            self.len_accepted_ttest.append( assigned_pseudo_labels.shape[0] ) 
            self.len_selected.append(  np.sum(self.num_augmented_per_class) )
        

        # remove the selected data from unlabelled data
        self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(labels_satisfied_threshold), 0)
        
        return X,y

    def label_assignment_and_post_processing(self, pseudo_labels_prob,X,y, current_iter=0,upper_threshold=None):
        """
        Given the threshold, we perform label assignment and post-processing

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


        if upper_threshold is None:
            upper_threshold=self.upper_threshold

        if 'CSA' in self.algorithm_name: # if using CSA, we dont use the upper threshold
            upper_threshold=0
            
        assigned_pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass)).astype(int)

        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass): # loop over each class

            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        

            temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > upper_threshold )[0]   
            labels_satisfied_threshold=idx_sorted[temp_idx]

            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold = labels_satisfied_threshold[:MaxPseudoPoint[cc]] 
            assigned_pseudo_labels[labels_satisfied_threshold, cc]=1

        if self.verbose:
            print("MaxPseudoPoint",MaxPseudoPoint)
        
        return self.post_processing_and_augmentation(assigned_pseudo_labels,X,y)

    

    def get_number_of_labels(self,y):
        """
        # given the label y, return the number of classes

        Args:
            y: label vector (for singlelabel) or matrix (for multilabel)

        Output:
            number of classes or number of labels
        """
        

        if self.IsMultiLabel==False:
            return len(np.unique(y))
        else:
            return y.shape[1]
        


    def get_max_pseudo_point(self,fraction_of_class, current_iter):
        """
        We select more points at the begining and less at later stage

        Args:
            fraction_of_class: vector of the frequency of points per class
            current_iter: current iteration  0,1,2...T
        Output:
            number_of_max_pseudo_points: scalar
        """ 
        
        LinearRamp= [(self.num_iters-ii)/self.num_iters for ii in range(self.num_iters)]
        SumLinearRamp=np.sum(LinearRamp)
        
        fraction_iter= (self.num_iters-current_iter) / (self.num_iters*SumLinearRamp)
        MaxPseudoPoint=fraction_iter*fraction_of_class*self.FractionAllocatedLabel*len(self.unlabelled_data)

        return np.int(np.ceil(MaxPseudoPoint))

        
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
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y)

            # evaluate_performance the performance on test set after Fit the model given the data
            self.evaluate_performance()
            
            # Predictive probability on the unlabeled data
            pseudo_labels_prob=self.get_predictive_prob_for_unlabelled_data(self.model)

            X,y=self.label_assignment_and_post_processing(pseudo_labels_prob,X,y,current_iter)

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " no training data ", len(y))
         
            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return
                
        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance()
        
    # def predict(self, X):
    #     return self.model.predict(X)
    
    # def predict_proba(self, X):
    #     return self.model.predict_proba(X)
    
    # def decision_function(self, X):
    #     return self.model.decision_function(X)