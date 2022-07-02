# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:19:22 2021

@author: Vu Nguyen
"""
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
import sklearn
    
class pseudo_labeling_iterative(object):
    # using all data points for pseudo-labels
    # iteratively take different percentage until all unlabelled data points are taken
    
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
                 fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        
        self.name="pseudo_labeling_iterative"
        self.x_test=x_test # N x d
        self.y_test=y_test # N x L whereL is the number of classes
        self.nClass=y_test.shape[1]
       
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

        # create a list of all the indices 
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))      
        
        self.selected_unlabelled_index=[]
        print("no of unlabelled data:",unlabelled_data.shape[0], "\t no of test data:",x_test.shape[0])

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)
        
        self.test_hl=[]
        self.test_mr=[]
        self.test_prec=[]
        
        self.ce_difference_list=[]
        self.MaxPseudoPoint_per_Class=20
        self.FractionAllocatedLabel=fraction_allocation
        self.num_XGB_models=9
        

        # for uncertainty estimation
        # generate multiple models
        # each XGBoost is a multilabel classifier
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
           
                mychoice=random.choice(params[key])
                #print(mychoice)
            
                param_list[tt][key]=mychoice
                param_list[tt]['verbosity'] = 0
                param_list[tt]['silent'] = 1
                param_list[tt]['seed'] = tt
                
            #self.XGBmodels_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)
            self.XGBmodels_list[tt]= MultiOutputClassifier(XGBClassifier(**param_list[tt],use_label_encoder=False))
    
    def data_uncertainty(self,pred):
    # pred [nModel, nPoint, nClass]
    
        ent=np.zeros((pred.shape[0],pred.shape[1]))
        for mm in range(pred.shape[0]):
            ent[mm,:]= self.entropy_prediction(pred[mm,:,:])
    
        return np.mean(ent,axis=0)
    
    def entropy_prediction(self,ave_pred):
        # pred [nPoint, nClass]
        
        ent=[0]*ave_pred.shape[0]
        
        for ii in range(ave_pred.shape[0]):
            ent[ii]= - np.sum( ave_pred[ii,:]*np.log(ave_pred[ii,:]))            
        return np.asarray(ent)
    
    def total_entropy(self,pred):#total uncertainty
        ave_pred=np.mean(pred,axis=0) # average over model
        total_uncertainty=self.entropy_prediction(ave_pred)
        return total_uncertainty
    
    
    def total_variance(self,pred):
        # [nModel, nPoint, nClass]
        std_pred = np.std( pred, axis=0) # std over model
        
        total_std = np.sum(std_pred, axis=1)
        
        return total_std
    
    
    def knowledge_uncertainty(self,pred):
        
        total_uncertainty=self.total_uncertainty(pred)
    
        data_uncertainty=self.data_uncertainty(pred)
    
        knowledge_uncertainty = total_uncertainty-data_uncertainty
        return knowledge_uncertainty


    def evaluate(self):
      

        y_test_pred = self.model.predict(self.x_test)
        
        
        # Match Ratio
        mr=accuracy_score(y_test_pred, self.y_test)*100
        
        #01Loss = np.any(self.y_test != y_test_pred, axis=1).mean()
        #Haming Loss
        hl=sklearn.metrics.hamming_loss(self.y_test, y_test_pred)*100

        # Precision
        prec=sklearn.metrics.precision_score(self.y_test, y_test_pred,average='samples')*100
        
        print('+++++Precision {:.2f}'.format(prec))
        
        self.test_hl +=[hl]
        self.test_mr +=[mr]
        self.test_prec +=[prec]
    
    def get_prob_at_max_class(self,pseudo_labels_prob):
        max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
        for jj in range(pseudo_labels_prob.shape[0]): 
            idxMax=np.argmax(pseudo_labels_prob[jj,:])
            max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
        return max_prob_matrix
    

    def set_ot_regularizer(self,nRow,nCol):
        
        if nRow/nCol>300:
            return 1
        elif nRow/nCol>200:
            return 0.5
        elif nRow/nCol>100:
            return 0.1
        else:
            return 0.01
        
    def get_max_pseudo_point(self,fraction_of_class, current_iter): # more at the begining and less at later stage
        #MaxPseudoPoint=fraction_of_class*np.int(self.FractionAllocatedLabel*len(self.unlabelled_data)/(self.num_iters*self.nClass))
        
        LinearRamp= [(self.num_iters-ii)/self.num_iters for ii in range(self.num_iters)]
        SumLinearRamp=np.sum(LinearRamp)
        
        fraction_iter= (self.num_iters-current_iter) / (self.num_iters*SumLinearRamp)
        MaxPseudoPoint=fraction_iter*fraction_of_class*self.FractionAllocatedLabel*len(self.unlabelled_data)
        
        return np.int(np.ceil(MaxPseudoPoint))
    
    def assignment_and_post_processing_CSA(self, assignment_matrix,pseudo_labels_prob,X,y, 
                                           current_iter=0,upper_threshold=None):
        
        
        #go over each row (data point), only keep the argmax prob            
        assignment_matrix = self.get_prob_at_max_class(assignment_matrix)
        
        if upper_threshold is None:
            upper_threshold=self.upper_threshold
            
        pseudo_labels=np.zeros((assignment_matrix.shape[0],self.nClass)).astype(int)
        #augmented_idx=[]

        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass):

            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( assignment_matrix[:,cc])[::-1] # decreasing        
        
            temp_idx1 = np.where( assignment_matrix[idx_sorted,cc] > 0 )[0]
            temp_idx2 = np.where( pseudo_labels_prob[idx_sorted[temp_idx1],cc] > 0.5)[0] 
            
            labels_within_threshold=idx_sorted[temp_idx2][:MaxPseudoPoint[cc]]

            pseudo_labels[labels_within_threshold, cc]=1

        print("MaxPseudoPoint",MaxPseudoPoint)
        
        temp=np.sum(pseudo_labels,axis=1)            
        labels_within_threshold = np.where(temp>0)[0]
        
        self.num_augmented_per_class.append( np.sum(pseudo_labels,axis=0).astype(int) )
        
        if len(labels_within_threshold) == 0: # no point is selected
            return X,y
            
        self.selected_unlabelled_index += labels_within_threshold.tolist()

        X = np.vstack((self.unlabelled_data[labels_within_threshold,:], X))
        y = np.vstack((pseudo_labels[labels_within_threshold,:], np.array(y)))
    
            
        # remove the selected data from unlabelled data
        #self.unlabelled_data = np.delete(self.unlabelled_data_master, np.unique(self.selected_unlabelled_index), 0)
        self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(labels_within_threshold), 0)
        
        return X,y

    def assignment_and_post_processing(self, max_prob_matrix,X,y, current_iter=0,upper_threshold=None):
        
        if upper_threshold is None:
            upper_threshold=self.upper_threshold
            
        pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass)).astype(int)

        MaxPseudoPoint=[0]*self.nClass
        for cc in range(self.nClass):

            MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
            
            idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        

            temp_idx = np.where(max_prob_matrix[idx_sorted,cc] > upper_threshold )[0]   
             
            pseudo_labels[idx_sorted[temp_idx[:MaxPseudoPoint[cc]]], cc]=1

        print("MaxPseudoPoint",MaxPseudoPoint)
        
        temp=np.sum(pseudo_labels,axis=1)            
        labels_within_threshold = np.where(temp>0)[0]
        
        self.num_augmented_per_class.append( np.sum(pseudo_labels,axis=0).astype(int) )
        
        if len(labels_within_threshold) == 0: # no point is selected
            return X,y
            
        self.selected_unlabelled_index += labels_within_threshold.tolist()

        X = np.vstack((self.unlabelled_data[labels_within_threshold,:], X))
        y = np.vstack((pseudo_labels[labels_within_threshold,:], np.array(y)))
    
            
        # remove the selected data from unlabelled data
        self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(labels_within_threshold), 0)
        
        return X,y
        
    def fit(self, X, y):
        """
        Perform pseudo labelling        
        X: train features
        y: train targets
        """
        print("===================",self.name)


        #self.num_augmented_per_class=[0]*self.nClass
        self.num_augmented_per_class=[]
        
        label_frequency = np.sum( y, axis=0)
        print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        self.label_frequency=label_frequency/np.sum(label_frequency)
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y)

            self.evaluate()
            
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            pseudo_labels_prob=np.asarray(pseudo_labels_prob).T
            pseudo_labels_prob=pseudo_labels_prob[1,:,:]
            
            num_points = pseudo_labels_prob.shape[0]

            #go over each row (data point), only keep the argmax prob
            max_prob=[0]*num_points
            
            max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            for jj in range(num_points): 
                idxMax=np.argmax(pseudo_labels_prob[jj,:])
                
                max_prob_matrix[jj,idxMax]=pseudo_labels_prob[jj,idxMax]
                max_prob[jj]=pseudo_labels_prob[jj,idxMax]
        
            X,y=self.assignment_and_post_processing(max_prob_matrix,X,y,current_iter)
            
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc
                
        # evaluate at the last iteration for reporting purpose
        self.evaluate()

        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def decision_function(self, X):
        return self.model.decision_function(X)

    
class flexmatch_multilabel(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
                 fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
        self.name="flex"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,fraction_of_class,current_iter):
        return super().get_max_pseudo_point(fraction_of_class,current_iter)
    def fit(self, X, y):
        print("===================",self.name)
     
        self.num_augmented_per_class=[]

        label_frequency = np.sum( y, axis=0)
        print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        self.label_frequency=label_frequency/np.sum(label_frequency)
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y)
            
            self.evaluate()

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            pseudo_labels_prob=np.asarray(pseudo_labels_prob).T
            pseudo_labels_prob=pseudo_labels_prob[1,:,:]
            
            num_points=pseudo_labels_prob.shape[0]
            # remove the old augmented data
        
            print("n_points/n_classes ={:d}/{:d} = {:.2f} ".format(num_points,self.nClass,
                  num_points/self.nClass))
            
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
            
            print("self.upper_threshold",self.upper_threshold)
            
            pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass))
            MaxPseudoPoint=[0]*self.nClass
            for cc in range(self.nClass):
                class_upper_thresh=countVector_normalized[cc]*self.upper_threshold

                MaxPseudoPoint[cc]=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1] # decreasing        

                temp_idx = np.where(max_prob_matrix[idx_sorted,cc] >class_upper_thresh )[0]                
                pseudo_labels[idx_sorted[temp_idx[:MaxPseudoPoint[cc]]], cc]=1

            temp=np.sum(pseudo_labels,axis=1)            
            labels_within_threshold = np.where(temp>0)[0]
            self.num_augmented_per_class.append( np.sum(pseudo_labels,axis=0).astype(int) )

            self.selected_unlabelled_index += labels_within_threshold.tolist()

            X = np.vstack((self.unlabelled_data[labels_within_threshold,:], X))
            y = np.vstack((pseudo_labels[labels_within_threshold,:], np.array(y)))
            
            if np.sum(self.num_augmented_per_class)==0: # no point is selected
                return #self.test_acc
            
            # remove the selected data from unlabelled data
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(labels_within_threshold), 0)
            

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))

        # evaluate at the last iteration for reporting purpose
        self.evaluate()

    


class sla(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
             fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
        self.name="sla total uncertainty"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,fraction_of_class,current_iter):
        return super().get_max_pseudo_point(fraction_of_class,current_iter)
            
    def data_uncertainty(self,pred):
    # pred [nModel, nPoint, nClass]
    
        ent=np.zeros((pred.shape[0],pred.shape[1]))
        for mm in range(pred.shape[0]):
            ent[mm,:]= self.entropy_prediction(pred[mm,:,:])
    
        return np.mean(ent,axis=0)

    def entropy_prediction(self,ave_pred):
        # pred [nPoint, nClass]
        
        ent=[0]*ave_pred.shape[0]
        
        for ii in range(ave_pred.shape[0]):
            ent[ii]= - np.sum( ave_pred[ii,:]*np.log(ave_pred[ii,:]))            
        return np.asarray(ent)
    
    def total_uncertainty(self,pred):
        ave_pred=np.mean(pred,axis=0) # average over model
        total_uncertainty=self.entropy_prediction(ave_pred)
        return total_uncertainty
    
    
    def total_variance(self,pred):
        # [nModel, nPoint, nClass]
        std_pred = np.std( pred, axis=0) # std over model
        
        total_std = np.sum(std_pred, axis=1)
        
        return total_std
    
    
    def knowledge_uncertainty(self,pred):
        
        total_uncertainty=self.total_uncertainty(pred)

        data_uncertainty=self.data_uncertainty(pred)

        knowledge_uncertainty = total_uncertainty-data_uncertainty
        return knowledge_uncertainty
    
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)

    def fit(self, X, y):
        print("===================",self.name)
      
        self.num_augmented_per_class=[]
        
        # estimate the frequency
        #unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        label_frequency = np.sum( y, axis=0)
        #print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        #label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y)
            self.teacher_model.fit(X, y)
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            #pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            
            for tt in range(self.num_XGB_models):
                self.XGBmodels_list[tt].fit(X, y)
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                temp = self.XGBmodels_list[tt].predict_proba(self.unlabelled_data)
                temp=np.asarray(temp).T
                pseudo_labels_prob_list[tt]=temp[1,:,:]
                
                
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
            #vu=np.mean(pseudo_labels_prob_list,axis=0) # debug
        
            #print("pseudo_labels_prob_list.shape",pseudo_labels_prob_list.shape)
            # get argmax in each row and second best 2nd argmax in each row
            #idxargmax = np.argmax(pseudo_labels_prob_list, axis=0)
            temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            # idxargmax=temp[:,0]
            # idx2nd_argmax= temp[:,1]
            # get uncertainty estimation in argmax and 2nd argmax
            
            #kappa=1
            M=3
          
            #entropy_score=self.knowledge_uncertainty(pseudo_labels_prob_list)
            entropy_score=self.total_uncertainty(pseudo_labels_prob_list)
            #total_std= self.total_variance(pseudo_labels_prob_list)
            #total_std=1-total_std
            # divide np.log(self.nClass) to normalised [0,1]
            uncertainty=1-entropy_score/np.log(self.nClass) 
            #uncertainty=max(0,uncertainty)
            uncertainty=np.clip(uncertainty, a_min=0,a_max=np.max(uncertainty))
            
            # expand marginal distribution for rows (points)
            # lcb_ucb=np.asarray(lcb_ucb)
            # lcb_ucb=np.clip(lcb_ucb, a_min=0,a_max=np.max(lcb_ucb))
            
            
            # for numerical stability of OT, select the nonzero entry only
            idxNoneZero=np.where(uncertainty>0)[0]
            num_points= len(idxNoneZero)
            
            print("num_points Entropy",num_points, "num_points", len(self.unlabelled_data))
            if len(idxNoneZero)==0: # terminate if could not find any point satisfying constraints
                return #self.test_acc
            
            
            upper_b_per_class=self.label_frequency*1.1
            lower_b_per_class=self.label_frequency*0.9
            
            idxZeroFreq= np.where(self.label_frequency==0 )[0]
            idxNonZeroFreq=np.where(self.label_frequency>0 )[0]
            upper_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*1.1
            lower_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*0.9
            
            
            #row_marginal=lcb_ucb[idxNoneZero]
            row_marginal=M*np.ones(num_points)
            #row_marginal=M*uncertainty[idxNoneZero]
                        
            #temp=M*np.sum(uncertainty[idxNoneZero])*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            temp=M*num_points*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            row_marginal = np.append(row_marginal,temp)
            
            #regulariser=0.005
            #regulariser=self.set_ot_regularizer(num_points, self.nClass)*200
            # dataset 2
            regulariser=self.set_ot_regularizer(num_points, self.nClass)*5

            
            print("n_points/n_classes ={:d}/{:d} = {:.2f}; regularizer={:.2f}".format(num_points,self.nClass,
                  num_points/self.nClass,regulariser))
            
            
            C=1-pseudo_labels_prob # cost # expand Cost matrix
            C=C[idxNoneZero,:]

            C=np.vstack((C,np.ones((1,self.nClass))))
            C=np.hstack((C,np.ones((len(idxNoneZero)+1,1))))
            
            K=np.exp(-C/regulariser)
            
            # plt.figure(figsize=(20,5))
            # plt.imshow(K.T)
            
            
            
            # expand marginal dist for columns (class)
            #col_marginal = M*upper_b_per_class*np.sum(uncertainty[idxNoneZero])  # frequency of the class label
            col_marginal = M*upper_b_per_class*num_points  # frequency of the class label
            
            #temp=M*np.sum(uncertainty[idxNoneZero])*(1-np.sum(lower_b_per_class))
            temp=M*num_points*(1-np.sum(lower_b_per_class))

            col_marginal = np.append(col_marginal,temp)
            
            
            if col_marginal.any()<0:
                print("NEGATIVE col_marginal",col_marginal)
            
            if row_marginal.any()<0:
                print("NEGATIVE row_marginal",row_marginal)
                
            if np.abs( np.sum(col_marginal) - np.sum(row_marginal) ) > 0.1 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001",np.sum(col_marginal) ,np.sum(row_marginal))
            
            uu=np.ones( (num_points+1,))
            
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= col_marginal / np.dot(K.T, uu)
                uu= row_marginal / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp2= np.atleast_2d(uu).T*(K*vv.T)
            
            #temp2 = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
            
            assignment_matrix=np.zeros((pseudo_labels_prob.shape))
            assignment_matrix[idxNoneZero,:]=temp2[:-1,:-1]
 
            X,y=self.assignment_and_post_processing_CSA(assignment_matrix,pseudo_labels_prob,X,y,\
                                                        current_iter,upper_threshold=0)

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc
                        
        # evaluate at the last iteration for reporting purpose
        self.evaluate()
        
        
class csa(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
             fraction_allocation=0.5,num_iters=10,confidence_choice='variance',verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
        self.confidence_choice=confidence_choice
        self.name="csa prob >0.5" + " confidence_choice:" + confidence_choice
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,fraction_of_class,current_iter):
        return super().get_max_pseudo_point(fraction_of_class,current_iter)
 
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)

    def fit(self, X, y):
        print("===================",self.name)
      
        self.num_augmented_per_class=[]
        
        # estimate the frequency
        #unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        label_frequency = np.sum( y, axis=0)        

        self.label_frequency=label_frequency/np.sum(label_frequency)
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y)
            self.teacher_model.fit(X, y)
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            #pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            
            for tt in range(self.num_XGB_models):
                self.XGBmodels_list[tt].fit(X, y)
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                temp = self.XGBmodels_list[tt].predict_proba(self.unlabelled_data)
                temp=np.asarray(temp).T
                pseudo_labels_prob_list[tt]=temp[1,:,:]
                
                
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
        
            #print("pseudo_labels_prob_list.shape",pseudo_labels_prob_list.shape)
            # get argmax in each row and second best 2nd argmax in each row
            #temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            
            if self.confidence_choice=="variance":
                total_var= self.total_variance(pseudo_labels_prob_list)
                confidence = 1-total_var
            elif self.confidence_choice=="neg_variance":
                total_var= self.total_variance(pseudo_labels_prob_list)
                confidence = total_var
            elif self.confidence_choice=="entropy":
                total_ent= self.total_entropy(pseudo_labels_prob_list)
                # divide np.log(self.nClass) to normalised [0,1]
                confidence=1-np.asarray(total_ent)/np.log(self.nClass) 
            elif self.confidence_choice=="neg_entropy":
                total_ent= self.total_entropy(pseudo_labels_prob_list)
                # divide np.log(self.nClass) to normalised [0,1]
                confidence=np.asarray(total_ent)/np.log(self.nClass) 

            confidence=confidence-np.mean(confidence)
            confidence=np.clip(confidence, a_min=0,a_max=np.max(confidence))
            
            
            # for numerical stability of OT, select the nonzero entry only
            idxNoneZero=np.where(confidence>0)[0]
            num_points= len(idxNoneZero)
            
            print("num_points Entropy",num_points, "num_points", len(self.unlabelled_data))
            if len(idxNoneZero)==0: # terminate if could not find any point satisfying constraints
                return #self.test_acc
            
            
            upper_b_per_class=self.label_frequency*1.1
            lower_b_per_class=self.label_frequency*0.9
            
            idxZeroFreq= np.where(self.label_frequency==0 )[0]
            idxNonZeroFreq=np.where(self.label_frequency>0 )[0]
            upper_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*1.1
            lower_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*0.9
            
            
            #row_marginal=lcb_ucb[idxNoneZero]
            row_marginal=np.ones(num_points)
            #row_marginal=M*uncertainty[idxNoneZero]
                        
            #temp=M*np.sum(uncertainty[idxNoneZero])*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            temp=num_points*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            row_marginal = np.append(row_marginal,temp)
            
            #regulariser=0.005
            
            if self.nClass>20:
                regulariser=self.set_ot_regularizer(num_points, self.nClass)*5
            else:
                regulariser=self.set_ot_regularizer(num_points, self.nClass)*200

            
            print("n_points/n_classes ={:d}/{:d} = {:.2f}; regularizer={:.2f}".format(num_points,self.nClass,
                  num_points/self.nClass,regulariser))
            
            
            C=1-pseudo_labels_prob # cost # expand Cost matrix
            C=C[idxNoneZero,:]

            C=np.vstack((C,np.ones((1,self.nClass))))
            C=np.hstack((C,np.ones((len(idxNoneZero)+1,1))))
            
            K=np.exp(-C/regulariser)
            
            
            # expand marginal dist for columns (class)
            #col_marginal = M*upper_b_per_class*np.sum(uncertainty[idxNoneZero])  # frequency of the class label
            col_marginal = upper_b_per_class*num_points  # frequency of the class label
            
            #temp=M*np.sum(uncertainty[idxNoneZero])*(1-np.sum(lower_b_per_class))
            temp=num_points*(1-np.sum(lower_b_per_class))

            col_marginal = np.append(col_marginal,temp)
            
            
            if col_marginal.any()<0:
                print("NEGATIVE col_marginal",col_marginal)
            
            if row_marginal.any()<0:
                print("NEGATIVE row_marginal",row_marginal)
                
            if np.abs( np.sum(col_marginal) - np.sum(row_marginal) ) > 0.1 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001",np.sum(col_marginal) ,np.sum(row_marginal))
            
            uu=np.ones( (num_points+1,))
            
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= col_marginal / np.dot(K.T, uu)
                uu= row_marginal / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp2= np.atleast_2d(uu).T*(K*vv.T)
            
            #temp2 = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
            
            assignment_matrix=np.zeros((pseudo_labels_prob.shape))
            assignment_matrix[idxNoneZero,:]=temp2[:-1,:-1]

            X,y=self.assignment_and_post_processing_CSA(assignment_matrix,pseudo_labels_prob,X,y,\
                                                        current_iter,upper_threshold=0)

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc
                        
        # evaluate at the last iteration for reporting purpose
        self.evaluate()




class sla_noconfidence(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
             fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
        self.name="sla marginal one no confidence"
        
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,fraction_of_class,current_iter):
        return super().get_max_pseudo_point(fraction_of_class,current_iter)
 
    
    def set_ot_regularizer(self,nRow,nCol):
        return super().set_ot_regularizer(nRow,nCol)

    def fit(self, X, y):
        print("===================",self.name)
      
        self.num_augmented_per_class=[]
        
        # estimate the frequency
        #unique, label_frequency = np.unique( y[np.sum(self.num_augmented_per_class):], return_counts=True)
        label_frequency = np.sum( y, axis=0)
        #print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        #label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

        self.label_frequency=label_frequency/np.sum(label_frequency)
        #label_frequency=np.ones( self.nClass )*1.0/self.nClass
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            self.selected_unlabelled_index=[]

            # Fit to data
            self.model.fit(X, y)
            self.teacher_model.fit(X, y)
            
            self.evaluate()
            
            num_points=self.unlabelled_data.shape[0]

            # estimate prob using unlabelled data
            #pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            
            for tt in range(self.num_XGB_models):
                self.XGBmodels_list[tt].fit(X, y)
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                temp = self.XGBmodels_list[tt].predict_proba(self.unlabelled_data)
                temp=np.asarray(temp).T
                pseudo_labels_prob_list[tt]=temp[1,:,:]
                
                
            pseudo_labels_prob_list=np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob= np.mean(pseudo_labels_prob_list,axis=0)
            #vu=np.mean(pseudo_labels_prob_list,axis=0) # debug
        
            #print("pseudo_labels_prob_list.shape",pseudo_labels_prob_list.shape)
            # get argmax in each row and second best 2nd argmax in each row
            #idxargmax = np.argmax(pseudo_labels_prob_list, axis=0)
            temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
            # idxargmax=temp[:,0]
            # idx2nd_argmax= temp[:,1]
            # get uncertainty estimation in argmax and 2nd argmax
            
            #kappa=1
            M=1
        
            upper_b_per_class=self.label_frequency*1.1
            lower_b_per_class=self.label_frequency*0.9
            
            idxZeroFreq= np.where(self.label_frequency==0 )[0]
            idxNonZeroFreq=np.where(self.label_frequency>0 )[0]
            upper_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*1.1
            lower_b_per_class[idxZeroFreq] = np.min(self.label_frequency[idxNonZeroFreq])*0.9
            
            
            row_marginal=M*np.ones(num_points)
            #row_marginal=M*uncertainty[idxNoneZero]
                        
            temp=M*num_points*(np.sum(upper_b_per_class)-np.sum(lower_b_per_class))
            row_marginal = np.append(row_marginal,temp)
            
            #regulariser=0.005
            regulariser=self.set_ot_regularizer(num_points, self.nClass)*200

            print("n_points/n_classes ={:d}/{:d} = {:.2f}; regularizer={:.2f}".format(num_points,self.nClass,
                  num_points/self.nClass,regulariser))
            
            
            C=1-pseudo_labels_prob # cost # expand Cost matrix

            C=np.vstack((C,np.zeros((1,self.nClass))))
            C=np.hstack((C,np.zeros((num_points+1,1))))
            
            K=np.exp(-C/regulariser)
            
            
            # expand marginal dist for columns (class)
            #col_marginal = M*upper_b_per_class*np.sum(uncertainty[idxNoneZero])  # frequency of the class label
            col_marginal = M*upper_b_per_class*num_points  # frequency of the class label
            
            #temp=M*np.sum(uncertainty[idxNoneZero])*(1-np.sum(lower_b_per_class))
            temp=M*num_points*(1-np.sum(lower_b_per_class))

            col_marginal = np.append(col_marginal,temp)
            
            
            if col_marginal.any()<0:
                print("NEGATIVE col_marginal",col_marginal)
            
            if row_marginal.any()<0:
                print("NEGATIVE row_marginal",row_marginal)
                
            if np.abs( np.sum(col_marginal) - np.sum(row_marginal) ) > 0.1 :
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001",np.sum(col_marginal) ,np.sum(row_marginal))
            
            uu=np.ones( (num_points+1,))
            
            #vv=np.ones( (pseudo_labels_prob.shape[1],))
            for jj in range(100):
                vv= col_marginal / np.dot(K.T, uu)
                uu= row_marginal / np.dot(K, vv)
                
            
            # recompute pseudo-label-prob from SLA
            temp2= np.atleast_2d(uu).T*(K*vv.T)
            
            #temp2 = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
            
            #pseudo_labels_prob=np.zeros((pseudo_labels_prob.shape))
            pseudo_labels_prob=temp2[:-1,:-1]
 
            #go over each row (data point), only keep the argmax prob            
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)

            X,y=self.assignment_and_post_processing(max_prob_matrix,X,y,current_iter,upper_threshold=0)

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc
                        
        # evaluate at the last iteration for reporting purpose
        self.evaluate()


#1. generate multiple XGB with different hyper

    

# UPS: https://arxiv.org/pdf/2101.06329.pdf
class UPS(pseudo_labeling_iterative):
    # adaptive thresholding
    
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
             fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
        
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
    def get_max_pseudo_point(self,fraction_of_class,current_iter):
        return super().get_max_pseudo_point(fraction_of_class,current_iter)
    def fit(self, X, y):
        
        print("====================",self.name)
        print("self.lower_threshold",self.lower_threshold)
     
        self.num_augmented_per_class=[]


        label_frequency = np.sum( y, axis=0)
        print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        self.label_frequency=label_frequency/np.sum(label_frequency)
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y)
            self.evaluate()
            
            num_points=len(self.unlabelled_data)

            for tt in range(self.num_XGB_models):
                self.XGBmodels_list[tt].fit(X, y)
                                                
            # estimate prob using unlabelled data
            pseudo_labels_prob_list=[0]*self.num_XGB_models
            for tt in range(self.num_XGB_models):
                temp = self.XGBmodels_list[tt].predict_proba(self.unlabelled_data)
                temp=np.asarray(temp).T
                pseudo_labels_prob_list[tt]=temp[1,:,:]
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
       
            # larger is betetr
       
            
            pseudo_labels=np.zeros((max_prob_matrix.shape[0],self.nClass)).astype(int)
            for cc in range(self.nClass):

                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1]# decreasing        
                
                temp_idx1 = np.where( max_prob_matrix[idx_sorted,cc] > self.upper_threshold )[0]
                temp_idx2 = np.where( uncertainty_rows[idx_sorted[temp_idx1],cc] < self.lower_threshold)[0]
                
                labels_within_threshold_cc=idx_sorted[temp_idx2][:MaxPseudoPoint]
                
                
                #temp_idx =np.intersect1d( idx_sorted[temp_idx1] , idx_sorted[temp_idx2])
                
                pseudo_labels[labels_within_threshold_cc, cc]=1

            temp=np.sum(pseudo_labels,axis=1)            
            labels_within_threshold = np.where(temp>0)[0]
            
            self.num_augmented_per_class.append( np.sum(pseudo_labels,axis=0).astype(int) )

            self.selected_unlabelled_index += labels_within_threshold.tolist()

            X = np.vstack((self.unlabelled_data[labels_within_threshold,:], X))
            y = np.vstack((pseudo_labels[labels_within_threshold,:], np.array(y)))
            
            # remove the selected data from unlabelled data
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(labels_within_threshold), 0)
            
            

            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))

            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc

            
        # evaluate at the last iteration for reporting purpose
        self.evaluate()
        
        
class sinkhorn_original(pseudo_labeling_iterative):
    # adaptive thresholding
    def __init__(self, model, unlabelled_data, x_test,y_test,upper_threshold = 0.8, lower_threshold = 0.2, 
           fraction_allocation=0.5,num_iters=10,verbose = False,datasetName=None):
        super().__init__(model, unlabelled_data, x_test,y_test,upper_threshold,lower_threshold ,fraction_allocation,num_iters,verbose)
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

        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
        
        
        self.num_augmented_per_class=[]
        
        label_frequency = np.sum( y, axis=0)
        print("==label_frequency w/o adjustment", np.round(label_frequency,3))
        
        #label_frequency=label_frequency/np.sum(label_frequency)+np.ones( self.nClass )*1.0/self.nClass

        self.label_frequency=label_frequency/np.sum(label_frequency)
        print("==label_frequency", np.round(self.label_frequency,3))
        
        
        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):
            
            # Fit to data
            self.model.fit(X, y)
            
            self.evaluate()
            

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            pseudo_labels_prob=np.asarray(pseudo_labels_prob).T
            pseudo_labels_prob=pseudo_labels_prob[1,:,:]
    
    
            num_points=pseudo_labels_prob.shape[0]
            # remove the old augmented data
        
            print("n_points/n_classes ={:d}/{:d} = {:.2f} ".format(num_points,self.nClass,
                  num_points/self.nClass))
            
            
            dist_points=np.ones( num_points+1 )
            
            #dist_points=dist_points/np.max(dist_points)
            
            regulariser=self.set_ot_regularizer(num_points,self.nClass)
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

            X,y=self.assignment_and_post_processing_CSA(assignment_matrix,pseudo_labels_prob,X,y,\
                                                        current_iter,upper_threshold=0)
            
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class[-1], " no training data ", len(y))
            if np.sum(self.num_augmented_per_class)==0:
                return #self.test_acc
                        
        # evaluate at the last iteration for reporting purpose
        self.evaluate()  
    