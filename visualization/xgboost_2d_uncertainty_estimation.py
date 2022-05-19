# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:46:08 2021

@author: Vu Nguyen
"""


import numpy as np
from matplotlib import pyplot as plt
#from xgboost import XGBClassifier
import xgboost as xgb
import random
from scipy import stats

# generate 2d data and visualize


# two gaussian distribution
np.random.seed(1)

mu1=[-0.3,1.2]
cov1=[[0.3,0.5],[0.5,2.8]]

data1=np.random.multivariate_normal(mu1, cov1, 30)
ydata1=[0]*data1.shape[0]


mu2=[2.5,2.8]
cov2=[[1.,0.4],[-0.4,1.]]

data2=np.random.multivariate_normal(mu2, cov2, 20)
ydata2=[1]*data2.shape[0]


mu3=[2.5,0.2]
cov3=[[0.7,0.5],[-0.5,0.8]]

data3=np.random.multivariate_normal(mu3, cov3, 20)
ydata3=[2]*data3.shape[0]



# mu4=[0.5,-0.2]
# cov4=[[0.5,0.5],[-0.5,0.5]]

# data4=np.random.multivariate_normal(mu4, cov4, 20)
# ydata4=[3]*data4.shape[0]

# ======================================= plot
fig=plt.figure()

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("True Data")


# ======================================= train XGB
X_train=np.vstack((data1,data2,data3))
y_train=np.asarray( ydata1+ydata2+ydata3)

x1_to_plot = np.linspace(-1.5, 4.5, 30)
x2_to_plot = np.linspace(-1.5, 4.7, 30)
x1_to_plot,x2_to_plot = np.meshgrid(x1_to_plot,x2_to_plot)

X_test= np.array([x1_to_plot.flatten(),x2_to_plot.flatten()]).T

model = xgb.XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)


num_points=X_train.shape[0]
nClass=len(np.unique(y_train))

y_pred = model.predict(X_train)


# idx1=np.where(y_pred==0)[0]
# idx2=np.where(y_pred==1)[0]

# plt.figure()

# plt.scatter(X_train[idx1,0],X_train[idx1,1],color='r')
# plt.scatter(X_train[idx2,0],X_train[idx2,1],color='b')

# plt.title("XGB Estimated Data")



# create multiple XGBmodel

params = { 'max_depth': np.arange(3, 20, 18).astype(int),
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.05),
           'colsample_bytree': np.arange(0.4, 1.0, 0.05),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.05),
           'n_estimators': [100, 200, 300, 500, 600, 700, 1000]}

num_models=30

# number of classes
C =len(np.unique(y_train))

param_list=[0]*num_models
for tt in range(num_models):
    
    param_list[tt]={}
    
    for key in params.keys():
        #print(key)
        # select value
        
        mychoice=random.choice(params[key])
        #print(mychoice)
    
        param_list[tt][key]=mychoice
        
    #param_list[tt]['silent']=1
    #param_list[tt]['verbose']=0
    
y_pred=np.zeros((num_models,X_test.shape[0],C ))
# train and predict        
for tt in range(num_models):
    
    #print(tt, param_list[tt])
    model = xgb.XGBClassifier(**param_list[tt],use_label_encoder=False)
    model.fit(X_train, y_train)
    
    temp=model.predict_proba(X_test)
    #y_pred[tt,:]=temp[:,0]
    y_pred[tt,:,:]=temp
#    y_pred[tt,:] = model.predict(X_test)

pseudo_labels_prob=np.mean(y_pred,axis=0)
temp=np.argsort(-pseudo_labels_prob,axis=1) # decreasing
idxargmax=temp[:,0]
idx2nd_argmax= temp[:,1]

num_points=pseudo_labels_prob.shape[0]

uncertainty_rows_argmax=[0]*num_points
uncertainty_rows_arg2ndmax=[0]*num_points

total_var=[0]*num_points

for jj in range(num_points):
    
    for cc in range(nClass):
        total_var[jj] += np.std( y_pred[:,jj,cc] )
    
    idxmax =idxargmax[jj]
    idx2ndmax=idx2nd_argmax[jj] 

    uncertainty_rows_argmax[jj]=np.std(y_pred[:,jj,idxmax  ])
    uncertainty_rows_arg2ndmax[jj]=np.std(y_pred[:,jj,idx2ndmax])

total_var=np.asarray(total_var)

uncertainty_rows_argmax=np.asarray(uncertainty_rows_argmax)
uncertainty_rows_arg2ndmax=np.asarray(uncertainty_rows_arg2ndmax)


t_test=[0]*num_points
t_value=[0]*num_points

var_rows_argmax=[0]*num_points
var_rows_arg2ndmax=[0]*num_points

for jj in range(num_points):# go over each row (data points)

    idxmax =idxargmax[jj]
    idx2ndmax=idx2nd_argmax[jj] 
    
    var_rows_argmax[jj]=np.var(y_pred[:,jj,idxmax  ])
    var_rows_arg2ndmax[jj]=np.var(y_pred[:,jj,idx2ndmax])
   
    nominator=pseudo_labels_prob[jj, idxmax]-pseudo_labels_prob[jj, idx2ndmax]
    temp=(0.1 + var_rows_argmax[jj] + var_rows_arg2ndmax[jj]  )/num_models
    denominator=np.sqrt(temp)
    t_test[jj] = nominator/denominator
    
    # compute degree of freedom
    nominator = (var_rows_argmax[jj] + var_rows_arg2ndmax[jj])**2
    
    denominator= var_rows_argmax[jj]**2 + var_rows_arg2ndmax[jj]**2
    denominator=denominator/(num_models-1)
    dof=nominator/denominator

    t_value[jj]=stats.t.ppf(1-0.025, dof)
    
    t_test[jj]=t_test[jj]-t_value[jj]
                
t_test=np.asarray(t_test)
t_test=np.clip(t_test,0,np.max(t_test))

y_pred=np.asarray(y_pred)
#y_pred.reshape((X_test.shape[0],num_models))
y_uncertainty=np.std(y_pred,axis=0)
y_uncertainty=np.mean(y_uncertainty,axis=1)

def entropy_prediction(pred):
    # pred [nPoint, nClass]
    
    ent=[0]*pred.shape[0]
    
    for ii in range(pred.shape[0]):
        ent[ii]= - np.sum( pred[ii,:]*np.log(pred[ii,:]))
        #ent[ii]=ent[ii]/pred.shape[1]
        
    return np.asarray(ent)

def data_uncertainty(pred):
    # pred [nModel, nPoint, nClass]
    
    ent=np.zeros((pred.shape[0],pred.shape[1]))
    for mm in range(pred.shape[0]):
        ent[mm,:]= entropy_prediction(pred[mm,:,:])

    return np.mean(ent,axis=0)

# total uncertainty
ave_pred=np.mean(y_pred,axis=0) # average over model

y_total_uncertainty=entropy_prediction(ave_pred)

y_data_uncertainty=data_uncertainty(y_pred)

y_knowledge_uncertainty = y_total_uncertainty-y_data_uncertainty
# data uncertainty


#plt.scatter(X_test[:,0],X_test[:,1],c=y_uncertainty,cmap='Greens')
CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,y_knowledge_uncertainty.reshape(x1_to_plot.shape),
                          cmap='Blues',origin='lower',alpha=0.3)
fig.colorbar(CS_acq_mean, shrink=0.9)
plt.title("Knowledge Uncertainty")
fig.savefig('knowledge_uncertainty_example.pdf',bbox_inches='tight')




# ======================================= plot
fig=plt.figure()

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("Total Uncertainty")

CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,y_total_uncertainty.reshape(x1_to_plot.shape),
                          cmap='Blues',origin='lower',alpha=0.3)
fig.colorbar(CS_acq_mean, shrink=0.9)
fig.savefig('total_uncertainty_example.pdf',bbox_inches='tight')



# ======================================= plot
fig=plt.figure()

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("Data Uncertainty")

CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,y_data_uncertainty.reshape(x1_to_plot.shape),
                          cmap='Blues',origin='lower',alpha=0.3)
fig.colorbar(CS_acq_mean, shrink=0.9)
fig.savefig('data_uncertainty_example.pdf',bbox_inches='tight')


# ======================================= plot
fig=plt.figure(figsize=(6,3.2))

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("Confidence = 1 - Total Variance",fontsize=14)

output=1-total_var
CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,output.reshape(x1_to_plot.shape),
                          cmap='plasma',origin='lower',alpha=0.3)
#plt.xlabel("Dimension 2")
#plt.ylabel("Dimension 1")

plt.xticks([])
plt.yticks([])

fig.colorbar(CS_acq_mean, shrink=0.9)
fig.savefig('total_var_example.pdf',bbox_inches='tight')



# ======================================= plot
fig=plt.figure()

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("Total Variance + Total Uncertainty")

total=total_var+y_total_uncertainty
total=np.clip(total,np.min(total),1)
CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,total.reshape(x1_to_plot.shape),
                          cmap='plasma',origin='lower',alpha=0.3)#Blues
fig.colorbar(CS_acq_mean, shrink=0.9)
fig.savefig('total_var_uncertainty_example.pdf',bbox_inches='tight')





# ======================================= plot
fig=plt.figure(figsize=(6,3.2))

plt.scatter(data1[:,0],data1[:,1],color='r')

plt.scatter(data2[:,0],data2[:,1],color='b')
plt.scatter(data3[:,0],data3[:,1],color='k')
#plt.scatter(data4[:,0],data4[:,1],color='g')

plt.title("Confidence = T-test",fontsize=14)

CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,t_test.reshape(x1_to_plot.shape),
                          cmap='plasma',origin='lower',alpha=0.3)

plt.xticks([])
plt.yticks([])
#plt.xlabel("Dimension 2")
#plt.ylabel("Dimension 1")
fig.colorbar(CS_acq_mean, shrink=0.9)
fig.savefig('ttest_confidence_example.pdf',bbox_inches='tight')

# ======================================= plot
# fig=plt.figure()

# plt.scatter(data1[:,0],data1[:,1],color='r')

# plt.scatter(data2[:,0],data2[:,1],color='b')
# plt.scatter(data3[:,0],data3[:,1],color='k')

# plt.title("Std Uncertainty Argmax")

# CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,uncertainty_rows_argmax.reshape(x1_to_plot.shape),
#                           cmap='Blues',origin='lower',alpha=0.3)
# fig.colorbar(CS_acq_mean, shrink=0.9)





# ======================================= plot
# fig=plt.figure()

# plt.scatter(data1[:,0],data1[:,1],color='r')

# plt.scatter(data2[:,0],data2[:,1],color='b')
# plt.scatter(data3[:,0],data3[:,1],color='k')

# plt.title("Std Uncertainty Arg 2nd max")

# CS_acq_mean=plt.contourf(x1_to_plot,x2_to_plot,uncertainty_rows_arg2ndmax.reshape(x1_to_plot.shape),
#                           cmap='Blues',origin='lower',alpha=0.3)
# fig.colorbar(CS_acq_mean, shrink=0.9)