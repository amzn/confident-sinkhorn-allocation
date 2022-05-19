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
from scipy import stats

np.random.seed(1)

# create a random matrix

def make_subplot_fig(myaxis,assignment_mat, mytitle,ylabel=None,IsEmptyYTick=None):
    myaxis.imshow(assignment_mat)
    myaxis.yaxis.set_major_locator(MaxNLocator(integer=True))
    myaxis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if IsEmptyYTick:
        myaxis.set_yticks([], [])
    myaxis.set_title(mytitle)
    myaxis.set_xlabel('Class')
    myaxis.set_ylabel(ylabel)

    
def get_csa_assignment(CostMatrix,idxNoneZero,regulariser,M,rho):
        
    print("====================M=",M,"rho=",rho,"regularizer=",regulariser)
    nRow,nCol=CostMatrix.shape
        
    #regulariser=1/(10*nCol)
        
    C=CostMatrix[idxNoneZero,:]
    
    C=np.vstack((C,np.zeros((1,nCol))))
    C=np.hstack((C,np.zeros((len(idxNoneZero)+1,1))))
    
    K=np.exp(-C/regulariser)
    
    upper_bound_per_class=[ 1/nCol +0.02 ]*nCol
    lower_bound_per_class=[ 1/nCol - 0.02]*nCol
    
    upper_bound_per_class=np.asarray(upper_bound_per_class)
    
    num_points=len(idxNoneZero)

    row_marginal=M*np.ones(num_points)
    
    temp=M*num_points*rho*(np.sum(upper_bound_per_class)-np.sum(lower_bound_per_class))
    
    
    row_marginal = np.append(row_marginal,temp)
    row_marginal=np.round(row_marginal,3)
    
    col_marginal=rho*M*num_points*upper_bound_per_class  #1
    
    
    #temp=M*np.sum(lcb_ucb[idxNoneZero])*(1-rho*np.sum(lower_bound_per_class))
    temp=M*num_points*(1-rho*np.sum(lower_bound_per_class))
    col_marginal = np.append(col_marginal, temp )
    
    row_marginal=np.round(row_marginal,4)
    col_marginal=np.round(col_marginal,4)
    
    print("row_marginal",row_marginal)
    print("col_marginal",col_marginal)
    print("sum(col_marginal)",np.sum(col_marginal),"sum(row_marginal)", np.sum(row_marginal))
    #assignment = ot.sinkhorn( row_marginal , col_marginal, reg=regulariser, M=C )
    
    uu=np.ones( (len(idxNoneZero)+1,))
    
    print("sum lower bound", np.sum(lower_bound_per_class))
    print("sum upper bound", np.sum(upper_bound_per_class))
    
    # check convergence
    
    gap_row=[0]*200
    gap_col=[0]*200
    
    for jj in range(200):
        vv= col_marginal / np.dot(K.T, uu)
        uu= row_marginal / np.dot(K, vv)
        
        assignment= np.atleast_2d(uu).T*(K*vv.T)
    
        gap_row[jj] = np.mean( np.abs( row_marginal - np.sum(assignment, axis=1) ))
        gap_col[jj] = np.mean( np.abs( col_marginal - np.sum(assignment, axis=0) ))    
    
    assignment= np.atleast_2d(uu).T*(K*vv.T)
    assignment=np.round(assignment,4)
    
    print("tau",assignment[-1,-1])
    print("u",assignment[:,-1])
    print("v",assignment[-1,:])
    
    assignment=assignment[:-1,:-1]
    
        
    assignment_matrix=np.zeros((nRow,nCol))
    assignment_matrix[idxNoneZero,:]=assignment
    assignment_matrix=np.round(assignment_matrix,2)

    return assignment_matrix

def get_assignment_PL( probMatrix, threshold):
    nRow,nCol=probMatrix.shape
    assignment_pl=np.zeros((nRow,nCol))
    for cc in range(nCol):
        idx_sorted=  np.argsort( ProbMatrix_removed[:,cc])[::-1] # decreasing        
        temp_idx = np.where(ProbMatrix_removed[idx_sorted,cc] > 0.8)[0]       
        assignment_pl[idx_sorted[temp_idx], cc]=1
    return assignment_pl



nRow= 25
nCol=5

M=5

ProbMatrix_list=[0]*M
CostMatrix_list=[0]*M

ProbMatrix_list[0] = np.random.random((nRow,nCol))
for jj in range(nRow):
    idxMax=np.argmax(ProbMatrix_list[0][jj])
    ProbMatrix_list[0][jj,idxMax]=ProbMatrix_list[0][jj,idxMax]+0.4
#for ii in range( int(M/2)):
for ii in range( 1,M):
    ProbMatrix_list[ii] = ProbMatrix_list[0]
    mymin=np.min(ProbMatrix_list[ii])
    mymax=np.max(ProbMatrix_list[ii])
    
    #ProbMatrix_list[ii]=(ProbMatrix_list[ii]-mymin)/( mymax-mymin)
    CostMatrix_list[ii]=1-ProbMatrix_list[ii]


ProbMatrix_rand=[0]*M
for ii in range( M):
    ProbMatrix_rand[ii]=np.random.random((nRow,nCol))
    
    
#NegCostMatrix_clean=np.zeros((NegCostMatrix.shape))


ProbMatrix_list=np.asarray(ProbMatrix_list)
ProbMatrix= np.mean(ProbMatrix_list,axis=0)


CostMatrix_list=np.asarray(CostMatrix_list)
CostMatrix= np.mean(CostMatrix_list,axis=0)

num_points=ProbMatrix.shape[0]

temp=np.argsort(-ProbMatrix,axis=1) # decreasing
idxargmax=temp[:,0]
idx2nd_argmax= temp[:,1]

t_test=[0]*nRow
uncertainty_rows_argmax=[0]*nRow
uncertainty_rows_arg2ndmax=[0]*nRow
t_test_matrix=np.zeros((nRow,nCol))

var_rows_argmax=[0]*num_points
var_rows_arg2ndmax=[0]*num_points

#lcb_ucb=[0]*num_points
t_test=[0]*num_points
t_value=[0]*num_points

for jj in range(num_points):# go over each row (data points)

    idxmax =idxargmax[jj]
    idx2ndmax=idx2nd_argmax[jj] 
    
    var_rows_argmax[jj]=0.005*np.random.rand()
    var_rows_arg2ndmax[jj]=0.005*np.random.rand()

    nominator=ProbMatrix[jj, idxmax]-ProbMatrix[jj, idx2ndmax]
    temp=(0.1 + var_rows_argmax[jj] + var_rows_arg2ndmax[jj]  )/M
    denominator=np.sqrt(temp)
    t_test[jj] = nominator/denominator
    
    # compute degree of freedom
    nominator = (var_rows_argmax[jj] + var_rows_arg2ndmax[jj])**2
    
    denominator= var_rows_argmax[jj]**2 + var_rows_arg2ndmax[jj]**2
    denominator=denominator/(M-1)
    dof=nominator/denominator

    t_value[jj]=stats.t.ppf(1-0.025, dof)
    
    t_test[jj]=t_test[jj]-t_value[jj]

t_test=np.asarray(t_test)
t_test=np.clip(t_test,0,np.max(t_test))
t_test=np.reshape(t_test,(num_points,1))

#print(t_test)

t_test=np.asarray(t_test)
idxNoneZero=np.where(t_test>0)[0]

N=len(idxNoneZero)






fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 6, width_ratios=[1,1,1,0.2,1,1]) 

# fig, (prob_axis1,prob_axis2,prob_axis3,_,confidence_axis,_,prob2_axis)=plt.subplots(1, 7, 
#                                                     gridspec_kw={'width_ratios': [2,2,2,1,2,1,2]})
#fig.tight_layout()
#fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

#axis3d = fig.add_subplot(1, 2, 1, projection='3d')
# prob_axis1 = fig.add_subplot(1, 7, 1)
# prob_axis2 = fig.add_subplot(1, 7, 2)
# prob_axis3 = fig.add_subplot(1, 7, 3)
# confidence_axis = fig.add_subplot(1, 7, 5)
# prob2_axis = fig.add_subplot(1, 7,7)
prob_axis1 = plt.subplot(gs[0])
prob_axis2 = plt.subplot(gs[1])
#prob_axis3 = plt.subplot(gs[2])
confidence_axis = plt.subplot(gs[2])
#prob2_axis = plt.subplot(gs[4])
cost_axis = plt.subplot(gs[4])
csa_axis = plt.subplot(gs[5])



im=prob_axis1.imshow(ProbMatrix_rand[0])
prob_axis1.set_xlabel('Class')
prob_axis1.set_ylabel('Unlabelled Data')
prob_axis1.set_title(r"Prob $P_{\theta_1}$")
prob_axis1.yaxis.set_major_locator(MaxNLocator(integer=True))
prob_axis1.xaxis.set_major_locator(MaxNLocator(integer=True))
#fig.colorbar(im, ax=prob_axis1)




make_subplot_fig(prob_axis2,ProbMatrix_rand[1], r"Prob $P_{\theta_2}$",IsEmptyYTick=True)


#prob_axis2.yaxis.set_major_locator(MaxNLocator(integer=True))
prob_axis2.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.colorbar(im, ax=prob_axis2)



im=confidence_axis.imshow(t_test)
confidence_axis.set_title("T_test")
confidence_axis.set_xticks([], [])
confidence_axis.set_yticks([], [])

fig.colorbar(im, ax=confidence_axis)


ProbMatrix_removed=np.zeros((nRow,nCol))
ProbMatrix_removed[idxNoneZero,:]=1-CostMatrix[idxNoneZero,:]



CostMatrix_removed=1-ProbMatrix_removed

make_subplot_fig(cost_axis,CostMatrix_removed, r"Cost $C=1-\bar{P}$",IsEmptyYTick=True)
fig.colorbar(im, ax=cost_axis)



assignment=get_csa_assignment(CostMatrix,idxNoneZero,regulariser=1/(20*nCol),M=1,rho=1)
make_subplot_fig(csa_axis,assignment,"CSA",IsEmptyYTick=True)



fig.tight_layout()
#fig.savefig("prob_ttest_step_visualization.pdf",bbox_inches="tight")


#===========================================================================
# visualization with different h value for multilabel
    
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1]) 


cost_axis = plt.subplot(gs[0])
csa_tau1 = plt.subplot(gs[1])
csa_tau2 = plt.subplot(gs[2])
csa_tau3 = plt.subplot(gs[3])

make_subplot_fig(cost_axis,CostMatrix, r"Cost $C=1-\bar{P}$",ylabel="Unlabelled Data")


assignment=get_csa_assignment(CostMatrix,idxNoneZero,regulariser=1/(3*nCol),M=1,rho=0.5)
make_subplot_fig(csa_tau1,assignment, r"CSA h=1")



assignment=get_csa_assignment(CostMatrix,idxNoneZero,regulariser=1/(3*nCol),M=4,rho=0.5)
make_subplot_fig(csa_tau2,assignment, r"CSA h=3")


assignment=get_csa_assignment(CostMatrix,idxNoneZero,regulariser=1/(3*nCol),M=6,rho=0.5)
make_subplot_fig(csa_tau3,assignment, r"CSA h=5")



fig.tight_layout()
fig.savefig("csa_viz_h_values.pdf",bbox_inches="tight")
