# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:38:44 2022

@author: Vu Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ot
from matplotlib.ticker import MaxNLocator

np.random.seed(1)

# create a random matrix

nRow= 25
nCol=5

nModel=20

NegCostMatrix= np.random.random((nRow,nCol))

NegCostMatrix_clean=np.zeros((NegCostMatrix.shape))

temp= np.random.random((nRow,nCol))

#NegCostMatrix[:,0:2]=NegCostMatrix[:,0:2]+0.7*temp[:,0:2]
#NegCostMatrix=(NegCostMatrix - np.min(NegCostMatrix) )/(np.max(NegCostMatrix)-np.min(NegCostMatrix))

CostMatrix=1-NegCostMatrix
#idxMax=np.argmax(CostMatrix,axis=1)

temp=np.argsort(-NegCostMatrix,axis=1) # decreasing
idxargmax=temp[:,0]
idx2nd_argmax= temp[:,1]


lcb_ucb1=[0]*nRow
lcb_ucb2=[0]*nRow
lcb_ucb=[0]*nRow
t_test=[0]*nRow
uncertainty_rows_argmax=[0]*nRow
uncertainty_rows_arg2ndmax=[0]*nRow
t_test_matrix=np.zeros((nRow,nCol))

lcb_ucb_matrix=np.zeros((nRow,nCol))
for ii in range(nRow):
    
    lcb_ucb1[ii]=NegCostMatrix[ii,idxargmax[ii]] - NegCostMatrix[ii,idx2nd_argmax[ii]] - 0.1
    lcb_ucb2[ii]=- NegCostMatrix[ii,idxargmax[ii]] + NegCostMatrix[ii,idx2nd_argmax[ii]] + 0.1
    
    lcb_ucb[ii]=max(lcb_ucb1[ii],lcb_ucb2[ii])
    lcb_ucb[ii]=max(0,lcb_ucb[ii])
    
    lcb_ucb_matrix[ii,idxargmax[ii]]=lcb_ucb1[ii]
    lcb_ucb_matrix[ii,idx2nd_argmax[ii]]=lcb_ucb2[ii]

    NegCostMatrix_clean[ii,idxargmax[ii]]=NegCostMatrix[ii,idxargmax[ii]]
    NegCostMatrix_clean[ii,idx2nd_argmax[ii]]=NegCostMatrix[ii,idx2nd_argmax[ii]]

    lcb_ucb_matrix[ii,idxargmax[ii]]=max(0,lcb_ucb_matrix[ii,idxargmax[ii]])
    lcb_ucb_matrix[ii,idx2nd_argmax[ii]]=max(0,lcb_ucb_matrix[ii,idx2nd_argmax[ii]])

    # uncertainty_rows_argmax[ii]=np.std(pseudo_labels_prob_list[:,ii,idxmax  ])
    # uncertainty_rows_arg2ndmax[ii]=np.std(pseudo_labels_prob_list[:,ii,idx2ndmax])
                
                
    # nominator=NegCostMatrix[ii, idxargmax[ii]]-NegCostMatrix[ii, idx2nd_argmax[ii]]
    # temp=( uncertainty_rows_argmax[ii]**2 + uncertainty_rows_arg2ndmax[ii]**2  )/self.num_XGB_models
    # denominator=np.sqrt(temp)
    # t_test[jj] = nominator/denominator
    # t_test_matrix[ii,idxargmax[ii]]=nominator/denominator
                
lcb_ucb=np.asarray(lcb_ucb)
idxNoneZero=np.where(lcb_ucb>0)[0]

N=len(idxNoneZero)

fig = plt.figure(figsize=(8, 5))
#fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

#axis3d = fig.add_subplot(1, 2, 1, projection='3d')
cost_axis = fig.add_subplot(1, 4, 1)
lcb_axis = fig.add_subplot(1, 4, 2)
assignment_prob_axis = fig.add_subplot(1, 4,3)
assignment_axis = fig.add_subplot(1, 4,4)
#subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

#acq2d = fig.add_subplot(1, 2, 2)

# multilabel
M=4

rho=1


#plt.figure()
#im=cost_axis.imshow(NegCostMatrix)
#cost_axis.set_title("Neg Cost Matrix")

im=cost_axis.imshow(CostMatrix)
cost_axis.set_xlabel('Class')
cost_axis.set_ylabel('Unlabelled Data')
cost_axis.set_title("Cost $C=1-P$")
cost_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
cost_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.colorbar(im, ax=cost_axis)

lcb_axis.imshow(lcb_ucb_matrix)
lcb_axis.set_title("lcb_ucb_matrix")

#regulariser=1/(10*nCol)

regulariser=0.01

C=-NegCostMatrix[idxNoneZero,:]

C=np.vstack((C,np.zeros((1,nCol))))
C=np.hstack((C,np.zeros((len(idxNoneZero)+1,1))))

K=np.exp(-C/regulariser)

upper_bound_per_class=[ 1/nCol +0.02 ]*nCol
lower_bound_per_class=[ 1/nCol - 0.02]*nCol

#upper_bound_per_class=[ 0.3]*nCol
upper_bound_per_class=np.asarray(upper_bound_per_class)


num_points=len(idxNoneZero)
#row_marginal=M*np.asarray(lcb_ucb[idxNoneZero])
row_marginal=M*np.ones(num_points)
#temp=rho*M*np.sum(lcb_ucb[idxNoneZero])*(np.sum(upper_bound_per_class)-np.sum(lower_bound_per_class))
temp=0.5*M*num_points*rho*(np.sum(upper_bound_per_class)-np.sum(lower_bound_per_class))


row_marginal = np.append(row_marginal,temp)
row_marginal=np.round(row_marginal,3)

col_marginal=0.5*rho*M*num_points*upper_bound_per_class  #1


#temp=M*np.sum(lcb_ucb[idxNoneZero])*(1-rho*np.sum(lower_bound_per_class))
temp=M*num_points*(1-0.5*rho*np.sum(lower_bound_per_class))
col_marginal = np.append(col_marginal, temp )

print("row_marginal",row_marginal)

print("col_marginal",col_marginal)
print("sum(col_marginal)",np.sum(col_marginal),"sum(row_marginal)", np.sum(row_marginal))
#assignment = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )

uu=np.ones( (len(idxNoneZero)+1,))

# check convergence

gap_row=[0]*300
gap_col=[0]*300

for jj in range(200):
    vv= col_marginal / np.dot(K.T, uu)
    uu= row_marginal / np.dot(K, vv)
    
    assignment= np.atleast_2d(uu).T*(K*vv.T)

    gap_row[jj] = np.mean( np.abs( row_marginal - np.sum(assignment, axis=1) ))
    gap_col[jj] = np.mean( np.abs( col_marginal - np.sum(assignment, axis=0) ))    

assignment= np.atleast_2d(uu).T*(K*vv.T)
assignment=np.round(assignment,5)

print("tau",assignment[-1,-1])
print("u",assignment[:,-1])
print("v",assignment[-1,:])

assignment=assignment[:-1,:-1]

idxMax=np.argmax(assignment,axis=1)

pseudo_labels_prob=np.zeros((nRow,nCol))
#pseudo_labels_prob[idxNoneZero,idxMax]=assignment[ np.arange( N ),idxMax]
#pseudo_labels_prob[idxNoneZero,:]=assignment

assignment_matrix=np.zeros((nRow,nCol))
assignment_matrix[idxNoneZero,:]=assignment

assignment_matrix=np.round(assignment_matrix,2)

im=assignment_prob_axis.imshow(assignment_matrix)
assignment_prob_axis.set_title("Assignment Matrix")
fig.colorbar(im, ax=assignment_prob_axis)


# get final assignment
for cc in range(nCol):
    idx_sorted=  np.argsort( assignment_matrix[:,cc])[::-1] # decreasing        

    temp_idx = np.where(assignment_matrix[idx_sorted,cc] > 0.01 )[0]       

    #MaxPseudoPoint=20         
    #pseudo_labels_prob[idx_sorted[temp_idx[:MaxPseudoPoint]], cc]=1
    pseudo_labels_prob[idx_sorted[temp_idx], cc]=1


assignment_axis.imshow(pseudo_labels_prob)
assignment_axis.set_title("Assignment")

print("np.sum(assignment_matrix,axis=0)",np.sum(assignment_matrix,axis=0))
print("np.sum(assignment_matrix,axis=1)",np.sum(assignment_matrix,axis=1))

print("np.sum(prob,axis=0)",np.sum(pseudo_labels_prob,axis=0))
print("np.sum(prob,axis=1)",np.sum(pseudo_labels_prob,axis=1))
print("mean np.sum(prob,axis=1)",np.mean(np.sum(pseudo_labels_prob,axis=1)))
assigned_idx=np.where(np.sum(pseudo_labels_prob,axis=1)>0)[0]
print("percent assignment", len(assigned_idx)*1.0/nRow)
print("sum lower bound", np.sum(lower_bound_per_class))
print("sum upper bound", np.sum(upper_bound_per_class))


fig.tight_layout()


# plt.figure()
# plt.plot(gap_row[5:],'b')
# plt.figure()
# plt.plot(gap_col[5:],'r')