# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:22:38 2017

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import datetime
import sys
import itertools
from sklearn import svm
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd

from IVreg_SVM_sim import IVreg_sim, IVreg_1stSVR_Est, IVreg_GMM_Est

setting_sim_temp={}
setting_sim_temp['n']=[100]
setting_sim_temp['mis']=[0]        
setting_sim_temp['simpoly']=[1]        
setting_sim_temp['estpoly']=[1]        
setting_sim_temp['alpha']=[0]
setting_sim_temp['N_inst']=[10]                
setting_sim_temp['N_char']=[1]

#setting_sim_temp['add_const_z']=[False]
#setting_sim_temp['add_const_x']=[False]
setting_sim_temp['var_eps_x']=[1.]
setting_sim_temp['var_eps_y']=[1.]
setting_sim_temp['var_eps_xy']=[.3]
setting_sim_temp['var_z']=[1.]
setting_sim_temp['mean_z']=[2.]

#setting_sim['endog'] = np.arange(setting_sim['N_char'])
setting_sim_temp['endog'] = [np.array([0])]
#setting_sim_temp['ivfunc']=['linear']
#setting_sim_temp['ivpoly_coeff'] =[np.array([1.]),np.array([1.,0.,-.52,0.,.016])]
setting_sim_temp['ivpoly'] = [3]
setting_sim_temp['ivpoly_interaction'] = [True]
       
setting_sim_temp['ivpoly_coeff'] = ['one']
#setting_sim_temp['ivpoly_coeff_mean'] = [10.]
#setting_sim_temp['ivpoly_coeff_var'] = [.3]
setting_sim_temp['ivpoly_coeff_mean_order'] = [[1.,.1,.01]]
setting_sim_temp['ivpoly_coeff_var_order'] = [[1.,.1,.01]]
setting_sim_temp['reg2nd_coeff'] =[[0.]]
setting_sims = model_selection.ParameterGrid(setting_sim_temp)

setting_est_temp = {}
setting_est_temp['add_const_x'] = [False]
setting_est_temp['iv_RC'] = [False]
setting_est_temp['endg_col'] = [ setting_sim_temp['endog'][0] ]
setting_est_svm1_temp = {}
setting_est_svm1_temp['C'] = [[1,10,10e2,10e3]]
setting_est_svm1_temp['gamma'] = [[1.,.1,10e-2,10e-3]]
setting_est_svm1_temp['n_cv'] = [3]
setting_ests = model_selection.ParameterGrid(setting_est_temp)
setting_est_svm1 = model_selection.ParameterGrid(setting_est_svm1_temp)

rep=100
B=100
bhat_est = np.zeros(rep)
bhat_boot = np.zeros([rep, B])
for setting_est in setting_ests:
    for setting_sim in setting_sims:
        for r in range(rep):
            i_setting_svm1=0
            for set_svm1 in setting_est_svm1:
                #Simulate
                sim = IVreg_sim(**setting_sim)
                sim.Sim()
                data = sim.Data
                data_svm1_rbf=data
                #Estimate
                est_svm1_rbf = IVreg_1stSVR_Est(Data=data_svm1_rbf, kernel='rbf',C=set_svm1['C'], gamma=set_svm1['gamma'], n_cv=set_svm1['n_cv'], **setting_est)
                est_svm1_rbf.Est()
                bhat_est[r] = est_svm1_rbf.bhat_svm1[0]
                #Bootstrap
                n = setting_sim['n']
                for b in range(B):
                    v = np.random.choice([-1.,1.],size=n,p=[.5,.5])
                    e_1st_b = (est_svm1_rbf.error_1ststage.flatten())*v
                    e_2nd_b = est_svm1_rbf.error_2ndstage*v
                    y_b = e_1st_b
                    
                    x_b = ( est_svm1_rbf.x_pred.flatten() )+e_2nd_b
                    x_b = x_b.reshape([-1,1])
                    data_b = {'x':x_b,'y':y_b, 'iv':data['iv']}
            
                    est_svm1_rbf_b = IVreg_1stSVR_Est(Data=data_b, kernel='rbf',C=set_svm1['C'], gamma=set_svm1['gamma'], n_cv=set_svm1['n_cv'], **setting_est)
                    est_svm1_rbf_b.Est()
                    bhat_boot[r,b] = est_svm1_rbf_b.bhat_svm1[0]

                i_setting_svm1=i_setting_svm1+1
