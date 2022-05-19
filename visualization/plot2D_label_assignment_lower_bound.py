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

np.random.seed(1)

# create a random matrix

nRow= 15
nCol=4

NegCostMatrix= np.random.random((nRow,nCol))


#idxMax=np.argmax(CostMatrix,axis=1)

temp=np.argsort(-NegCostMatrix,axis=1) # decreasing
idxargmax=temp[:,0]
idx2nd_argmax= temp[:,1]


lcb_ucb=[0]*nRow
lcb_ucb_matrix=np.zeros((nRow,nCol))
for ii in range(nRow):
    lcb_ucb_matrix[ii,idxargmax[ii]]=NegCostMatrix[ii,idxargmax[ii]] -NegCostMatrix[ii,idx2nd_argmax[ii]]- 0.1
    lcb_ucb[ii]=NegCostMatrix[ii,idxargmax[ii]] - NegCostMatrix[ii,idx2nd_argmax[ii]] - 0.1
    lcb_ucb[ii]=max(0,lcb_ucb[ii])
    lcb_ucb_matrix[ii,idxargmax[ii]]=max(0,lcb_ucb_matrix[ii,idxargmax[ii]])

lcb_ucb=np.asarray(lcb_ucb)
idxNoneZero=np.where(lcb_ucb>0)[0]

N=len(idxNoneZero)

fig = plt.figure(figsize=(8, 5))

#axis3d = fig.add_subplot(1, 2, 1, projection='3d')
cost_axis = fig.add_subplot(1, 4, 1)
lcb_axis = fig.add_subplot(1, 4, 2)
assignment_prob_axis = fig.add_subplot(1, 4,3)
assignment_axis = fig.add_subplot(1, 4,4)

#acq2d = fig.add_subplot(1, 2, 2)



#plt.figure()
im=cost_axis.imshow(NegCostMatrix)
cost_axis.set_title("Neg Cost Matrix")

#divider = make_axes_locatable(cost_axis)
#cax = divider.append_axes('right', size='10%', pad=0.1)
#fig.colorbar(im, cax=cax, orientation='vertical')


lcb_axis.imshow(lcb_ucb_matrix)
lcb_axis.set_title("lcb_ucb_matrix")

regulariser=0.2
C=np.exp(-NegCostMatrix[idxNoneZero,:]/regulariser)

C=np.vstack((C,np.ones((1,nCol))))
C=np.hstack((C,np.ones((len(idxNoneZero)+1,1))))

#upper_bound_per_class=[ nRow/nCol + 0.5]*nCol
upper_bound_per_class=[ 0.4]*nCol
upper_bound_per_class=np.asarray(upper_bound_per_class)

lower_bound_per_class=[ 0.1]*nCol
lower_bound_per_class=np.asarray(lower_bound_per_class)

print("upper",upper_bound_per_class,"lower",lower_bound_per_class)

# row marginal
dist_points=np.asarray(lcb_ucb[idxNoneZero])
dist_points = np.append(dist_points,N*(np.sum(lower_bound_per_class)-np.sum(upper_bound_per_class)))


#dist_labels=[np.sum(dist_points)/nCol]*nCol
#dist_labels=[np.sum(dist_points)/nCol]*nCol

dist_labels=N*lower_bound_per_class #+ np.mean(lcb_ucb[idxNoneZero],axis=0) #1
dist_labels=np.asarray(dist_labels)#-1.0/nCol

dist_labels = np.append(dist_labels,N*(np.mean(lcb_ucb[idxNoneZero])-np.sum(upper_bound_per_class)))

print("dist_labels",dist_labels)

print("dist_points",dist_points)

print("sum(dist_labels)",np.sum(dist_labels),"sum(dist_points)", np.sum(dist_points))
assignment = ot.sinkhorn( dist_points , dist_labels.T, reg=regulariser, M=C )

assignment=np.round(assignment,3)

omega=-assignment[-1,-1]
print("-omega",-omega)
uu=assignment[:-1,-1]
print("u",uu)
vv=-assignment[-1,:-1]
print("-v",-vv)

print("N",N)
assignment2=assignment[:-1,:-1]

idxMax=np.argmax(assignment2,axis=1)

pseudo_labels_prob=np.zeros((nRow,nCol))
pseudo_labels_prob[idxNoneZero,idxMax]=assignment2[ np.arange( N ),idxMax]
#pseudo_labels_prob[idxNoneZero,:]=assignment

assignment_matrix=np.zeros((nRow,nCol))
assignment_matrix[idxNoneZero,:]=assignment2


assignment_prob_axis.imshow(assignment_matrix)
assignment_prob_axis.set_title("Assignment Matrix")



assignment_axis.imshow(pseudo_labels_prob)
assignment_axis.set_title("Assignment")
