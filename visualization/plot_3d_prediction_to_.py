# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:38:11 2022

@author: Vu Nguyen
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator

np.random.seed(1)

# Create axis
axes = [5, 4, 10]
  
# Create Data
data = np.ones(axes, dtype=np.bool)
  
# Controll Tranperency
alpha = 0.9
  
# Control colour
colors = np.empty(axes + [4], dtype=np.float32)
  
colors[0] = [1, 0, 0, alpha]  # red
colors[1] = [0, 1, 0, alpha]  # green
colors[2] = [0, 0, 1, alpha]  # blue
colors[3] = [1, 1, 0, alpha]  # yellow
colors[4] = [1, 1, 1, alpha]  # grey
  
# Plot figure
fig = plt.figure(figsize=(14,4))
ax = fig.add_subplot(111, projection='3d')
  
# Voxels is used to customizations of
# the sizes, positions and colors.
ax.voxels(data, facecolors=colors, edgecolors='grey')


ax.tick_params(axis='both', which='major', pad=-3)
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-4)
ax.tick_params(axis='z', which='major', pad=-3)

ax.zaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))


ax.set_xlabel("N data points",fontsize=13,labelpad=-4)
ax.set_ylabel("K classes",fontsize=13,labelpad=-4)
ax.set_zlabel("M classifiers",fontsize=13,labelpad=-3)



#plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
#plt.rcParams['axes.titlepad'] = -4  # pad is in points...

ax.set_title(r"Prediction Probs $P_{M \times N \times K}$",fontsize=16,y=1.0, pad=0)
fig.savefig("3d_prediction_probs.pdf",bbox_inches="tight")


#plt.rcParams['axes.titlepad'] = 1  # pad is in points...


nRow=10
nCol=5


ProbMatrix_rand=np.random.random((nRow,nCol))
#ProbMatrix_rand2=np.random.random((nRow,nCol))

ProbMatrix_rand[:,0]=ProbMatrix_rand[:,0]-0.15
ProbMatrix_rand[:,4]=ProbMatrix_rand[:,4]-0.15
ProbMatrix_rand[:,1]=ProbMatrix_rand[:,1]-0.1
ProbMatrix_rand[:,2]=ProbMatrix_rand[:,2]*1.4+0.15

ProbMatrix_rand=np.clip(ProbMatrix_rand,a_min=0,a_max=1)


fig=plt.figure(figsize=(10,4.5))

gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
prob_axis1 = plt.subplot(gs[0])
prob_axis2 = plt.subplot(gs[1])
prob_axis3 = plt.subplot(gs[2])

im=prob_axis1.imshow(ProbMatrix_rand)
prob_axis1.set_xlabel('K classes',fontsize=16)
prob_axis1.set_ylabel('M classifiers',fontsize=16)
prob_axis1.set_title(r"$P_{M \times [i=0] \times K}$",fontsize=22)
prob_axis1.yaxis.set_major_locator(MaxNLocator(integer=True))
prob_axis1.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.colorbar(im, ax=prob_axis1)

#xrange=np.arange(nRow)
xrange=np.linspace(0,1,nRow*20)

#data = np.random.normal(0.5, 1, 20)

data=ProbMatrix_rand[:,2]

# Plotting the histogram.
prob_axis2.hist(data, bins=50, density=False, alpha=0.6, color='g')
#prob_axis2.bar(np.arange(nRow),data)

mu, std = norm.fit(data)
p = norm.pdf(xrange, mu, std)
prob_axis2.plot(xrange, p, 'k', linewidth=2,label="$\mathcal{N}(\mu_{\diamond},\sigma_{\diamond})$")

prob_axis2.set_title(r"Highest Score $P_{M \times [i=0,k=2]}$",fontsize=16)
prob_axis2.set_xlabel("Prediction Probs",fontsize=16)
#prob_axis2.set_ylabel("M classifiers")
prob_axis2.set_ylabel("Density",fontsize=16)
prob_axis2.legend(fontsize=18)





data=ProbMatrix_rand[:,1]
#prob_axis3.bar(np.arange(nRow),data)
prob_axis3.hist(data, bins=50, density=False, alpha=0.6, color='b')


mu, std = norm.fit(data)
p = norm.pdf(xrange, mu, std)
prob_axis3.plot(xrange, p, 'k', linewidth=2,label="$\mathcal{N}(\mu_{\oslash},\sigma_{\oslash})$")


prob_axis3.set_title(r"2nd-Highest $P_{M \times [i=0,k=1]}$",fontsize=16)
prob_axis3.set_xlabel("Prediction Probs",fontsize=16)
#prob_axis3.set_xlabel("M classifiers")
prob_axis3.set_ylabel("Density",fontsize=16)
prob_axis3.legend(fontsize=18)

fig.tight_layout()
fig.savefig("illustration_ttest_1d_histogram.pdf",bbox_inches="tight")

    
    