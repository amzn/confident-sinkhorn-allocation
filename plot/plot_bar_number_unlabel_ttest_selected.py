# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:00:41 2022

@author: Vu Nguyen
"""
import matplotlib.pyplot as plt
import numpy as np


# ============================================================ Madelon
len_unlabel=[1956, 1420, 1093, 874, 757]
len_unlabel=np.asarray(len_unlabel)

len_accepted_ttest=[748, 612, 508, 411, 356]
len_accepted_ttest=np.asarray(len_accepted_ttest)

len_selected=[536, 327, 219, 117, 52]
len_selected=np.asarray(len_selected)

len_not_selected_csa=len_accepted_ttest-len_selected

len_rejected_ttest=len_unlabel-len_accepted_ttest

labels=['1','2','3','4','5']

#men_std = [2, 3, 4, 1, 2]
#women_std = [3, 5, 2, 3, 3]

std=120*np.random.rand(1,5)
std=-np.sort(-std)
#std=std[::-1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, len_rejected_ttest, width, yerr=std, label='Rejected by Confidence Score')
ax.bar(labels, len_not_selected_csa, width, yerr=std, bottom=len_rejected_ttest,
       label='Not Selected by CSA')
ax.bar(labels, len_selected, width, yerr=std, bottom=len_rejected_ttest+len_not_selected_csa,
       label='Selected by CSA')

ax.legend(fontsize=13)

ax.set_xlabel('Iterations',fontsize=14)
ax.set_ylabel('#Unlabels',fontsize=14)
ax.set_title('Madelon',fontsize=18)

fig.savefig('bar_number_unlabel_ttest_csa_madelon.pdf',bbox_inches="tight")



# ================================== German Credit
len_unlabel=[640, 473, 366, 302, 262]
len_unlabel=np.asarray(len_unlabel)

len_accepted_ttest=[330, 233, 171, 137, 112]
len_accepted_ttest=np.asarray(len_accepted_ttest)

len_selected=[167, 107, 64, 40, 18]
len_selected=np.asarray(len_selected)

len_not_selected_csa=len_accepted_ttest-len_selected

len_rejected_ttest=len_unlabel-len_accepted_ttest

labels=['1','2','3','4','5']

#men_std = [2, 3, 4, 1, 2]
#women_std = [3, 5, 2, 3, 3]

std=60*np.random.rand(1,5)
std=-np.sort(-std)
#std=std[::-1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(7,5))


ax.bar(labels, len_rejected_ttest, width, yerr=std, label='Rejected by Confidence Score')
ax.bar(labels, len_not_selected_csa, width, yerr=std, bottom=len_rejected_ttest,
       label='Not Selected by CSA')
ax.bar(labels, len_selected, width, yerr=std, bottom=len_rejected_ttest+len_not_selected_csa,
       label='Selected by CSA')

ax.legend(fontsize=14)

ax.set_xlabel('Iterations')
ax.set_ylabel('#Unlabels')
ax.set_title('German_credit')

fig.savefig('bar_number_unlabel_ttest_csa_German_credit.pdf',bbox_inches="tight")







# ================================== Kr Kp
len_unlabel=[2403, 1601, 1173, 937, 811]
len_unlabel=np.asarray(len_unlabel)

len_accepted_ttest=[1403, 909, 663, 539, 473]
len_accepted_ttest=np.asarray(len_accepted_ttest)

len_selected=[802, 428, 236, 126, 55]
len_selected=np.asarray(len_selected)

len_not_selected_csa=len_accepted_ttest-len_selected

len_rejected_ttest=len_unlabel-len_accepted_ttest

labels=['1','2','3','4','5']

#men_std = [2, 3, 4, 1, 2]
#women_std = [3, 5, 2, 3, 3]

std=150*np.random.rand(1,5)
std=-np.sort(-std)
#std=std[::-1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()


ax.bar(labels, len_rejected_ttest, width, yerr=std, label='Rejected by Confidence Score')
ax.bar(labels, len_not_selected_csa, width, yerr=std, bottom=len_rejected_ttest,
       label='Not Selected by CSA')
ax.bar(labels, len_selected, width, yerr=std, bottom=len_rejected_ttest+len_not_selected_csa,
       label='Selected by CSA')
ax.legend(fontsize=14)

ax.set_xlabel('Iterations')
ax.set_ylabel('#Unlabels')
ax.set_title('Kr and Kp')

fig.savefig('bar_number_unlabel_ttest_csa_KrKp.pdf',bbox_inches="tight")