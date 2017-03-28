# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:14:46 2017

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import datetime
import sys
import os
import itertools
from sklearn import svm
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd




def Extract(results, coeff_ext, interaction_ext, N_inst_ext):
    n_seq = np.array([])
    ols_mean_bias_seq = np.array([])
    ols_mean_abs_seq = np.array([])
    ols_var_seq = np.array([])
    tsls0_mean_bias_seq = np.array([])
    tsls0_mean_abs_seq = np.array([])
    tsls0_var_seq = np.array([])
    tsls1_mean_bias_seq = np.array([])
    tsls1_mean_abs_seq = np.array([])
    tsls1_var_seq = np.array([])
    svm_mean_bias_seq = np.array([])
    svm_mean_abs_seq = np.array([])
    svm_var_seq = np.array([])
    for res1 in results:
        if np.all(res1['setting_sim']['ivpoly_coeff_mean_order'] == coeff_ext) and res1['setting_sim']['ivpoly_interaction'] == interaction_ext and res1['setting_sim']['N_inst'] == N_inst_ext:
            n_seq = np.append(n_seq, res1['setting_sim']['n'] )
            
            ols = res1['bhat_ols'].flatten()
            ols_mean_abs = np.mean( np.abs( ols-1. ) )
            ols_mean_bias = np.mean( ols-1. )
            ols_var = np.var(ols)
            ols_mean_bias_seq = np.append( ols_mean_bias_seq, ols_mean_bias )
            ols_mean_abs_seq = np.append( ols_mean_abs_seq, ols_mean_abs )
            ols_var_seq = np.append( ols_var_seq, ols_var )
            
            tsls0 = res1['bhat_2sls'][:,0,0]
            tsls0_mean_abs = np.mean( np.abs( tsls0-1. ) )
            tsls0_mean_bias = np.mean( tsls0-1. )
            tsls0_var = np.var(tsls0)
            tsls0_mean_bias_seq = np.append( tsls0_mean_bias_seq, tsls0_mean_bias )
            tsls0_mean_abs_seq = np.append( tsls0_mean_abs_seq, tsls0_mean_abs )
            tsls0_var_seq = np.append( tsls0_var_seq, tsls0_var )
            
            tsls1 = res1['bhat_2sls'][:,0,1]
            tsls1_mean_abs = np.mean( np.abs( tsls1-1. ) )
            tsls1_mean_bias = np.mean( tsls1-1. )
            tsls1_var = np.var(tsls1)
            tsls1_mean_bias_seq = np.append( tsls1_mean_bias_seq, tsls1_mean_bias )
            tsls1_mean_abs_seq = np.append( tsls1_mean_abs_seq, tsls1_mean_abs )
            tsls1_var_seq = np.append( tsls1_var_seq, tsls1_var )
            
            svm = res1['bhat_svm1_rbf'].flatten()
            svm_mean_abs = np.mean( np.abs( svm-1. ) )
            svm_mean_bias = np.mean( svm-1. )
            svm_var = np.var( svm )
            svm_mean_bias_seq = np.append( svm_mean_bias_seq, svm_mean_bias )
            svm_mean_abs_seq = np.append( svm_mean_abs_seq, svm_mean_abs )
            svm_var_seq = np.append( svm_var_seq, svm_var )
            
    return n_seq, ols_mean_bias_seq, ols_mean_abs_seq, ols_var_seq,\
             tsls0_mean_bias_seq, tsls0_mean_abs_seq, tsls0_var_seq,\
             tsls1_mean_bias_seq, tsls1_mean_abs_seq, tsls1_var_seq,\
             svm_mean_bias_seq, svm_mean_abs_seq, svm_var_seq

def Graph(n_seq,title, ols, tsls0,tsls1 , svm1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_seq, ols)
    ax.plot(n_seq, tsls0)
    ax.plot(n_seq, tsls1)
    ax.plot(n_seq, svm1)
    ax.legend(['OLS','2SLS0','2SLS1','SVR'])
    ax.set_title(title)
def unique_rows(A, return_index=False, return_inverse=False):
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
               return_index=return_index,
               return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')    
    
    
results1 = np.load('results_all1_Mar-26-2017.npy')
results2 = np.load('results_all2_Mar-26-2017.npy')
results3 = np.load('results_all3_Mar-26-2017.npy')
results4 = np.load('results_all4_Mar-26-2017.npy')
results = np.append( results1, results2)
results = np.append( results, results3)
results = np.append( results, results4)

i=0
for res in results:
    if i==0:
        coeff_tried = np.array(res['setting_sim']['ivpoly_coeff_mean_order'])
    else:
        coeff_tried = np.c_[ coeff_tried.T, res['setting_sim']['ivpoly_coeff_mean_order'] ].T
    i=i+1
coeff_tried = unique_rows(coeff_tried)


for coeff_ext in coeff_tried:
    interaction_ext = True
    N_inst_ext = 7
    n_seq, ols_mean_bias_seq, ols_mean_abs_seq, ols_var_seq,\
    tsls0_mean_bias_seq, tsls0_mean_abs_seq, tsls0_var_seq,\
    tsls1_mean_bias_seq, tsls1_mean_abs_seq, tsls1_var_seq,\
    svm_mean_bias_seq, svm_mean_abs_seq, svm_var_seq\
    =Extract(results, coeff_ext, interaction_ext, N_inst_ext)
    #Graph(n_seq,str(coeff_ext),ols_mean_bias_seq, tsls0_mean_bias_seq, svm_mean_bias_seq)

    Graph(n_seq,str(coeff_ext),ols_mean_abs_seq, tsls0_mean_abs_seq, tsls1_mean_abs_seq, svm_mean_abs_seq)

