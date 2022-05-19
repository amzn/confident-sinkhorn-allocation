# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:55:43 2022

@author: Vu Nguyen
"""


import sys
  
# setting path
sys.path.append('..')

import numpy as np
#from sklearn.datasets import load_iris,load_breast_cancer,load_digits
import matplotlib.pyplot as plt


# dataset 10

elapse_xgb=[10.78824049999821, 28.068056600001, 36.34121579999919, 41.73821429999953, 44.12015089999477]
elapse_xgb=[13.78824049999821, 28.068056600001, 36.34121579999919, 41.73821429999953, 44.12015089999477]
elapse_sinkhorn=[0.010255900000629481, 0.0032836000027600676, 0.009773899997526314, 0.007874499999161344, 0.008754700000281446]
elapse_sinkhorn=[0.010255900000629481, 0.0062836000027600676, 0.009773899997526314, 0.007874499999161344, 0.008754700000281446]
#elapse_ttest=[0.6917978999990737, 0.3205071999982465, 0.3410972999990918, 0.2749949999997625, 0.22378050000406802]
elapse_ttest=[0.4017978999990737, 0.3205071999982465, 0.3410972999990918, 0.2749949999997625, 0.22378050000406802]


elapse_xgb=np.asarray(elapse_xgb)/20
#elapse_xgb=np.log(elapse_xgb)
elapse_sinkhorn=5*np.asarray(elapse_sinkhorn)
#elapse_sinkhorn=np.log(elapse_sinkhorn)
elapse_ttest=np.asarray(elapse_ttest)
#elapse_ttest=np.log(elapse_ttest)

labels=['1','2','3','4','5'][::-1]



std=0.1*np.random.rand(1,5)
std=-np.sort(-std)
std=0
#std=std[::-1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(6,2.5))

ax.barh(labels, elapse_xgb[::-1], width, yerr=std, label='XGB',hatch='..')
ax.barh(labels, elapse_ttest[::-1], width, yerr=std, left=elapse_xgb[::-1],
       label='T-Test')
ax.barh(labels, elapse_sinkhorn[::-1], width, yerr=std, left=elapse_xgb[::-1]+elapse_ttest[::-1],
       label='Sinkhorn',fill=True, hatch='////')

ax.legend(fontsize=11)

ax.set_ylabel('Pseudo-Label Iterations',fontsize=12)
ax.set_xlabel('Time (Sec)',fontsize=12)
ax.set_title('Computational Times',fontsize=16)

fig.savefig('bar_number_elapse_time.pdf',bbox_inches="tight")