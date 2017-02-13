# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:54:27 2016

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

class IVreg_sim:
    '''
    def __init__(self, n=500, mis=0, endog=np.array([0]), simpoly=1, estpoly=1,\
                 alpha=0., N_inst=10, N_char=5,\
                 var_eps_x=1., var_eps_y=1., var_eps_xy=3.,\
                 mean_z=1., var_z=1., add_const_x=False, add_const_z=False,\
                 ivfunc='linear',ivpoly_coeff=np.array([1.,-1.,-1.]) ,savedata=0,\
                 iv_RC=False, n_ind=10, iv_RC_var=1.  #Panel Data
                 ):
    ''' 
    def __init__(self, n=500, mis=0, endog=np.array([0]), simpoly=1, estpoly=1,\
                 alpha=0., N_inst=10, N_char=5,\
                 var_eps_x=1., var_eps_y=1., var_eps_xy=3.,\
                 mean_z=1., var_z=1., add_const_x=True, add_const_z=True,\
                 ivpoly=1,ivpoly_coeff='random',ivpoly_coeff_var=1.,ivpoly_coeff_mean=0., savedata=0,\
                 iv_RC=False, n_ind=10, iv_RC_var=1.  #Panel Data
                 ):
        self.iv_RC=iv_RC
        self.n = n
        if self.iv_RC==True:
            self.n_ind = n_ind
            self.n_period = self.n/self.n_ind
            self.ind = np.repeat( np.arange(self.n_ind), self.n_period )
            self.period = np.tile( np.arange(self.n_period), self.n_ind )
            self.iv_RC_var = iv_RC_var
        
        self.mis=mis        
        self.endog = endog
        
        #self.ivpoly = ivpoly_coeff.size     
        self.ivpoly = ivpoly
        self.simpoly = simpoly        
        self.estpoly = estpoly        
        self.alpha=alpha
        self.N_inst = N_inst                
        self.N_char = N_char
        self.var_eps_x = var_eps_x
        self.var_eps_y = var_eps_y
        self.var_eps_xy = var_eps_xy
        
        self.mean_z = mean_z
        self.var_z = var_z
        
        #self.ivfunc=ivfunc
       
        self.savedata=savedata
        
        self.add_const_x = add_const_x
        self.add_const_z = add_const_z
        
        #self.ivpoly_coeff=np.repeat(ivpoly_coeff, self.N_inst)
        self.ivpoly_coeff = ivpoly_coeff
        if ivpoly_coeff=='random':
            self.ivpoly_coeff_var = ivpoly_coeff_var
            self.ivpoly_coeff_mean = ivpoly_coeff_mean
            
        
    def Sim(self):
        if self.mis==2:
            alpha = self.alpha
        elif self.mis==1:
            alpha = self.alpha/float(self.n)
        elif self.mis==0:
            alpha = 0.
                    
        z = self.mean_z+np.random.normal(size=[self.n,self.N_inst])*np.sqrt( self.var_z)
        eps_x = np.random.normal(size=[self.n, self.N_char]) * np.sqrt(self.var_eps_x)
        eps_xy = np.random.normal(size=[self.n, self.N_char])* np.sqrt(self.var_eps_xy)
        eps_y = np.random.normal(size=self.n)* np.sqrt(self.var_eps_y)
        
        '''
        z_poly=z        
        for i in range(2,self.ivpoly+1):
            z_poly=np.c_[z_poly, z**i]
        self.z_poly=z_poly
        if self.ivfunc=='quad':
            z_poly = self.QuadPoly_vec(z_poly)
        if self.add_const_z:
            z_poly  = np.c_[np.ones(self.n), z_poly]
        '''
        genpoly = preprocessing.PolynomialFeatures(degree=self.ivpoly, include_bias=self.add_const_z)
        z_poly = genpoly.fit_transform(z)
        self.N_inst_total = z_poly.shape[1]
        '''
        if self.ivpoly_coeff is None:
            x_data =  np.repeat( np.sum(z_poly,axis=1),  self.N_char).reshape([self.n, self.N_char])+eps_x
            #x_data =  np.repeat( np.mean(z_poly,axis=1),  self.N_char).reshape([self.n, self.N_char])+eps_x
        elif self.ivpoly_coeff is not None:
            x_data =  np.repeat( np.dot(z_poly,self.ivpoly_coeff),  self.N_char ).reshape([self.n, self.N_char]) +eps_x         
        x_data=x_data+eps_xy*np.in1d(np.arange(self.N_char), self.endog)
        '''
        if self.ivpoly_coeff is 'random':
            self.ivpoly_coeff = self.ivpoly_coeff_mean + np.random.normal(size=self.N_inst_total)*np.sqrt(self.ivpoly_coeff_var)
        if self.ivpoly_coeff is 'one':
            self.ivpoly_coeff = np.ones(self.N_inst_total)
        x_data =  np.repeat( np.dot(z_poly,self.ivpoly_coeff),  self.N_char ).reshape([self.n, self.N_char]) +eps_x
            
        if self.iv_RC:
            RC = np.random.multivariate_normal(np.zeros(z_poly.shape[1]), np.sqrt(self.iv_RC_var)*np.eye( z_poly.shape[1] ),  size=self.n_ind)
            self.RC1=RC
            RC = RC[self.ind,:]
            self.RC2=RC
            self.RC3=np.repeat( np.sum(RC*z_poly,axis=1), self.N_char).reshape([self.n,-1])
            x_data = x_data + np.repeat( np.sum(RC*z_poly,axis=1), self.N_char).reshape([self.n,-1])

        x_poly=x_data        
        for i in range(2,self.simpoly+1):
            x_poly=np.c_[x_poly, x_data**i]
        if self.add_const_x:
            x_poly = np.c_[np.ones(self.n),x_poly]

        y = np.sum(x_poly,axis=1).flatten() +alpha*z[:,-1]

        y=y+eps_y+np.mean(eps_xy,axis=1)
            
        x=x_data        
        for i in range(2,self.estpoly+1):
            x=np.c_[x, x_data**i]
            
        self.Data = {'x':x,'y':y,'iv':z}
        if self.iv_RC==True:
            dummy = self.CreateDummy( self.ind )
            self.Data = {'x':x,'y':y,'iv':z,'dummy':dummy}
    '''
    def QuadPoly_vec(self,vec_vec):
        if vec_vec.ndim==1:
            f = self.QuadPoly(vec_vec)
            return f
            
        rep = vec_vec.shape[0]
        n = vec_vec.shape[1]
        i = np.tril_indices(n)[0]
        j = np.tril_indices(n)[1]
        n_out = i.shape[0]    
        ii = np.tile(i,rep)+np.repeat(np.arange(rep)*n, n_out)
        jj = np.tile(j,rep)
        
        b = np.tile(vec_vec, n).reshape([n*rep,n])
        c = np.repeat(vec_vec, n).reshape([n*rep,n])
        d = b*c
        e = d[ii,jj].reshape([rep,n_out])
        f = np.c_[vec_vec, e]
        return f
    '''
    
    def CreateDummy(self,groupid, sparse=0):
        nobs = groupid.size
        id_list = np.unique(groupid)
        id_num = id_list.size    
        if sparse==0:
            groupid_dum = np.zeros([nobs,id_num])
        elif sparse==1:
            groupid_dum = scipy.sparse.lil_matrix((nobs,id_num))
            
        for i in range(id_num):
            a = (groupid==id_list[i]).repeat(id_num).reshape([nobs,id_num])
            b = (id_list==id_list[i]).repeat(nobs).reshape([id_num,nobs]).T
            c = a*b
            groupid_dum[c] = 1        
        return groupid_dum

        
        
class IVreg_1stSVR_Est:
    def __init__(self, Data, add_const_x, iv_RC, endg_col, kernel, n_cv=3,C=[10,100,1000,10000],gamma= [1.,.1,.01,.001,.0001], **kwargs):
        self.x = Data['x']            
        self.y = Data['y']
        self.iv = Data['iv']                
        self.n, self.n_char = self.x.shape
        self.n_iv = self.iv.shape[1]
        if add_const_x==True:
            self.x = np.c_[np.ones(self.n), self.x]
            self.iv = np.c_[np.ones(self.n), self.iv]
        if iv_RC==True:
            self.iv = np.c_[self.iv, Data['dummy']]
            
        
        self.endg_col = endg_col
        self.exog_col = np.delete( np.arange(self.n_char), self.endg_col)
        
        self.IV = np.c_[ self.x[:,self.exog_col],self.iv ]
        self.n_IV = self.IV.shape[1]

        if kernel=='rbf':
            self.param_grid = [\
            {'C': C, 'gamma': gamma, 'kernel': [ kernel ]},]
        if kernel=='linear':
            self.param_grid = [\
            {'C': C, 'kernel': [ kernel ]},]
        self.n_cv=n_cv
            
    def Est(self):
        x_scaler=preprocessing.StandardScaler()
        #x_scaled=x_scaler.fit_transform(data['x'])
        x_scaled=self.x        
        y_scaler=preprocessing.StandardScaler()
        #y_scaled=y_scaler.fit_transform(data['y'].reshape(-1, 1))        
        y_scaled=self.y
        IV_scaler=preprocessing.StandardScaler()
        #IV_scaled=self.IV        
        IV_scaled=IV_scaler.fit_transform(self.IV)
         
        self.bhat_svm1=np.zeros(self.n_char)

        x_pred = np.zeros_like(x_scaled)
        x_pred[:] = x_scaled[:]
        for j in self.endg_col:
            svr_rbf = svm.SVR()
            
            gridsearch = model_selection.GridSearchCV(svr_rbf,param_grid=self.param_grid,refit=True, cv=self.n_cv)
            gridsearch.fit(IV_scaled,x_scaled[:,j])
            x_pred[:,j]=gridsearch.predict(IV_scaled)  
        print(gridsearch.best_estimator_)
        lr2=linear_model.LinearRegression(fit_intercept=False)
        lr2.fit(x_pred,y_scaled)
        self.bhat_svm1 = lr2.coef_
        self.EstResult = {'bhat':self.bhat_svm1}

        
        
        
        

        
        
        
class IVreg_2stageSVR_Est:
    def __init__(self, Data, add_const_x, endg_col, kernel, **kwargs):
        self.x = Data['x']            
        self.y = Data['y']
        self.iv = Data['iv']                
        self.n, self.n_char = self.x.shape
        self.n_iv = self.iv.shape[1]
        if add_const_x==True:
            self.x = np.c_[np.ones(self.n), self.x]
            self.iv = np.c_[np.ones(self.n), self.iv]
        
        self.endg_col = endg_col
        self.exog_col = np.delete( np.arange(self.n_char), self.endg_col)
        
        self.IV = np.c_[ self.x[:,self.exog_col],self.iv ]
        self.n_IV = self.IV.shape[1]
        self.param_grid = [\
        {'C': [10,100,1000,10000], 'gamma': [.1,.01,.001,.0001], 'kernel': [ kernel ]},]
        
    def Est_1st(self):
        x_scaler=preprocessing.StandardScaler()
        #x_scaled=x_scaler.fit_transform(data['x'])
        x_scaled=self.x        
        IV_scaler=preprocessing.StandardScaler()
        #IV_scaled=self.IV        
        IV_scaled=IV_scaler.fit_transform(self.IV)
         
        self.bhat_svm1=np.zeros(self.n_char)

        x_pred = np.zeros_like(x_scaled)
        x_pred[:] = x_scaled[:]
        for j in self.endg_col:
            svr_rbf = svm.SVR()
            gridsearch = model_selection.GridSearchCV(svr_rbf,param_grid=self.param_grid,refit=True)
            gridsearch.fit(IV_scaled,x_scaled[:,j])
            x_pred[:,j]=gridsearch.predict(IV_scaled)
        self.x_pred = x_pred
        
    def Est_2nd(self):    
        x_pred_scaler=preprocessing.StandardScaler()
        #x_pred_scaled=x_scaler.fit_transform(self.x_pred)
        x_pred_scaled=self.x_pred        

        y_scaler=preprocessing.StandardScaler()
        #y_scaled=y_scaler.fit_transform(data['y'].reshape(-1, 1))        
        y_scaled=self.y
        svr_rbf = svm.SVR()
        self.gridsearch_2nd = model_selection.GridSearchCV(svr_rbf,param_grid=self.param_grid,refit=True)
        self.gridsearch_2nd.fit(x_pred_scaled,y_scaled)

    def Est(self):
        self.Est_1st()
        self.Est_2nd()
        
    def SimulateMEAtMean(self):
        self.y_pred_ME = np.zeros([50,self.n_char])
        self.lin = np.zeros([50,self.n_char])
        x_mean = np.mean(self.x,axis=0)
        x_std = np.std(self.x,axis=0)
        for i in range(self.n_char):
            self.lin[:,i] = np.linspace(x_mean[i]-x_std[i], x_mean[i]+x_std[i],num=50 )
            x_me = np.tile(x_mean,50).reshape([50, self.n_char])
            x_me[:,i] = self.lin[:,i]
            self.y_pred_ME[:,i] = self.gridsearch_2nd.predict(x_me)
            

class IVreg_GMM_Est:
    
    def __init__(self, Data, add_const_x, endg_col, reg_type, weight_type,iv_poly=1, **kwargs):
    #def __init__(self, Data, **kwargs):
        self.x = Data['x']            
        self.y = Data['y']
        genpoly = preprocessing.PolynomialFeatures(degree=iv_poly,include_bias=False)
        self.iv = genpoly.fit_transform( Data['iv'] )                
        self.n, self.n_char = self.x.shape
        self.n_iv = self.iv.shape[1]
        if add_const_x==True:
            self.x = np.c_[np.ones(self.n), self.x]
            self.iv = np.c_[np.ones(self.n), self.iv]
        
        self.endg_col = endg_col
        self.exog_col = np.delete( np.arange(self.n_char), self.endg_col)
        
        self.IV = np.c_[ self.x[:,self.exog_col],self.iv ]
        #self.IV = self.iv 
        self.n_IV = self.IV.shape[1]
        #self.temp = np.dot( self.IV.T, self.IV )            
        
        self.reg_type = reg_type
        self.weight_type = weight_type
        if self.weight_type =='identity':
            self.weight = np.eye(self.n_IV).astype(float)
        elif self.weight_type =='invA':
            temp =  np.dot( self.IV.T, self.IV )
            self.weight = np.linalg.inv( temp)
        
    def Est(self,robust=False):
        if self.reg_type=='2sls':
            bhat, f = self.ivreg_2sls(self.x, self.y, self.IV, self.weight)
        if self.reg_type=='2tepGMM':
            bhat, f = self.ivreg_2stepGMM(self.x, self.y, self.IV, self.weight)
        AsyVar = self.ivreg_2sls_AsyVar(self.x,self.xhat,self.y,bhat,robust=robust)
            
        self.EstResult = {'bhat':bhat, 'W':self.weight, 'AsyVar':AsyVar}
        return bhat, f
        
    def ivreg_2sls(self,x,y,z,invA=None):
        n = y.size
        self.xhat=np.dot( np.linalg.inv(np.dot(z.T,z) ),np.dot(z.T,x) ) 
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        temp1 = np.dot(x.T,z)
        temp2 = np.dot(y.T,z)
        temp3 = np.dot(np.dot(temp1,invA),temp1.T) #x'z(z'z)^{-1}z'x
        temp4 = np.dot(np.dot(temp1,invA),temp2.T) #x'z(z'z)^{-1}z'y
        if temp3.size==1 or temp4.size==1:
            bhat = temp4/temp3
        else:
            try:
                bhat = np.linalg.solve(temp3,temp4)
            except np.linalg.LinAlgError:
                self.flag_SingularMatrix=1
                print("singular matrix")
                return
        gmmresid = y - np.dot(x, bhat)
        temp5=np.dot(gmmresid.T, z)
        f=np.dot(np.dot(temp5/n, invA),(temp5.T)/n)
        del x,y,z,invA
        return bhat,f
        
    def ivreg_2sls_AsyVar(self,x,xhat,y,bhat,adjust=True,robust=False):
        n,k=x.shape
        ehat = y - np.dot(x,bhat)
        xhat2inv = np.linalg.inv(np.dot(xhat.T,xhat) )
        if not robust:
            sigsq = np.mean(ehat**2)
            if adjust:
                sigsq = sigsq*n/(n-k)
            var = sigsq* xhat2inv
        if robust:
            Sig = np.diag(ehat**2)
            mid = np.dot( np.dot(xhat.T, Sig), xhat)
            var = np.dot( np.dot(xhat2inv, mid), xhat2inv)
        return var
        

    def ivreg_2stepGMM(self,x,y,z,invA=None):
        n = y.size
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        bhat_1st,f1 = self.ivreg_2sls(x,y,z,invA)
        eps = np.array([y - np.dot(x, bhat_1st)]).T
        W2 = (1./n) * np.dot( np.dot(z.T, eps), np.dot(eps.T, z) )
        bhat_2nd, f2 = self.ivreg_2sls(x,y,z,invA=W2)
        return bhat_2nd,f2,W2
        
        



if __name__=='__main__':
    

    setting_sim_temp={}
    setting_sim_temp['n']=[500, 2000]
    setting_sim_temp['mis']=[0]        
    setting_sim_temp['simpoly']=[1]        
    setting_sim_temp['estpoly']=[1]        
    setting_sim_temp['alpha']=[0]
    setting_sim_temp['N_inst']=[10]                
    setting_sim_temp['N_char']=[5]
    
    #setting_sim_temp['add_const_z']=[False]
    #setting_sim_temp['add_const_x']=[False]
    setting_sim_temp['var_eps_x']=[1.]
    setting_sim_temp['var_eps_y']=[1.]
    setting_sim_temp['var_eps_xy']=[3.]
    setting_sim_temp['var_z']=[1.]
    setting_sim_temp['mean_z']=[1.]

    #setting_sim['endog'] = np.arange(setting_sim['N_char'])
    setting_sim_temp['endog'] = [np.array([0])]
    #setting_sim_temp['ivfunc']=['linear']
    #setting_sim_temp['ivpoly_coeff'] =[np.array([1.]),np.array([1.,0.,-.52,0.,.016])]
    setting_sim_temp['ivpoly'] = [3]           
    setting_sim_temp['ivpoly_coeff'] = ['random']       
    #setting_sim['ivpoly_coeff'] =np.array([1.])       
    #Panel Data
    setting_sim_temp['iv_RC']=[True]
    setting_sim_temp['n_ind']=[10]
    setting_sim_temp['iv_RC_var']=[1.]
    

    setting_est_temp={}
    setting_est_temp['reg_type'] = ['2sls']
    setting_est_temp['weight_type'] = ['invA']
    setting_est_temp['add_const_x'] = [False]
    setting_est_temp['iv_RC'] = [False]
    setting_est_temp['endg_col'] = [ setting_sim_temp['endog'][0] ]
    
    setting_sims = model_selection.ParameterGrid(setting_sim_temp)
    setting_ests = model_selection.ParameterGrid(setting_est_temp)

    n_settings = len(list(setting_sims)) * len(list(setting_ests))
    
    #setting for each method
    setting_est_ols_temp = {}
    setting_est_ols_temp['poly'] = [1]
    setting_est_2sls_temp = {}
    setting_est_2sls_temp['poly'] = [1]
    setting_est_svm1_temp = {}
    setting_est_svm1_temp['C'] = [[10],[10e4],[10,10e2,10e3,10e4]]
    setting_est_svm1_temp['gamma'] = [[1.],[10e-4],[1.,.1,10e-2,10e-3,10e-4]]
    setting_est_svm1_temp['n_cv'] = [3]
    
    setting_est_ols = model_selection.ParameterGrid(setting_est_ols_temp)
    setting_est_2sls = model_selection.ParameterGrid(setting_est_2sls_temp)
    setting_est_svm1 = model_selection.ParameterGrid(setting_est_svm1_temp)
    n_setting_ols = len(list(setting_est_ols))
    n_setting_2sls = len(list(setting_est_2sls))
    n_setting_svm1 = len(list(setting_est_svm1))
    
    '''
    for i in setting_sims:
        print(i)
    sys.exit()
    '''
    results_all=[]
    rep = 2
    start=time.time()
    for setting_est in setting_ests:
        for setting_sim in setting_sims:
            
            '''
            bhat_ols=np.zeros([rep,setting_sim['N_char'] + setting_est['add_const_x'] ])
            bhat_2sls=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'] ])
            bhat_svm1_linear=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'] ])
            bhat_svm1_rbf=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'] ])
            bhat_svm1_rbf_cv5=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'] ])
            '''
            '''
            lin_svm2 = np.zeros([rep, 50 ,setting_sim['N_char']  + setting_est['add_const_x'] ])
            y_svm2 = np.zeros([rep, 50 ,setting_sim['N_char']  + setting_est['add_const_x'] ])
            bhat_svm2=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'] ])
            '''
            bhat_ols=np.zeros([rep,setting_sim['N_char'] + setting_est['add_const_x'],  n_setting_ols])
            bhat_2sls=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'], n_setting_2sls ])
            bhat_svm1_rbf=np.zeros([rep,setting_sim['N_char']  + setting_est['add_const_x'], n_setting_svm1 ])
            
            for i in range(rep):
                print('rep: '+str(i))
                sim = IVreg_sim(**setting_sim)
                sim.Sim()
                data = sim.Data
        
                #OLS
                i_setting_ols=0
                for set_ols in setting_est_ols:
                    lr1=linear_model.LinearRegression(fit_intercept=setting_est['add_const_x'])
                    
                    lr1.fit(data['x'],data['y'])
                    if setting_est['add_const_x']:
                        bhat_ols[i,:,i_setting_ols] = np.insert(lr1.coef_,0, lr1.intercept_)
                    if setting_est['add_const_x']==False:
                        bhat_ols[i,:,i_setting_ols] = lr1.coef_
                    i_setting_ols=i_setting_ols+1
                
                #2sls
                data_2sls = data
                i_setting_2sls=0                
                for set_2sls in setting_est_2sls:
                    est_2sls = IVreg_GMM_Est(data_2sls, iv_poly=set_2sls['poly'], **setting_est)
                    est_2sls.Est()
                    bhat_2sls[i,:,i_setting_2sls] = est_2sls.EstResult['bhat']
                    i_setting_2sls=i_setting_2sls+1
                
                #SVM first stage + linear regression second stage
                ##BRF kernel
                data_svm1_rbf=data
                i_setting_svm1=0
                for set_svm1 in setting_est_svm1:
                    
                    est_svm1_rbf = IVreg_1stSVR_Est(data_svm1_rbf, kernel='rbf',C=set_svm1['C'], gamma=set_svm1['gamma'], n_cv=set_svm1['n_cv'], **setting_est)
                    est_svm1_rbf.Est()
                    bhat_svm1_rbf[i,:,i_setting_svm1] = est_svm1_rbf.EstResult['bhat']
                    i_setting_svm1=i_setting_svm1+1

                '''
                #2stage SVM
                ##RBF kernel
                data_svm2=data
                if setting_est['add_const_x']==True:
                    data_svm2['x'] = np.c_[ np.ones(setting_sim['n'] ), data_svm2['x']  ]
                est_svm2 = IVreg_2stageSVR_Est(data_svm2, kernel='rbf', **setting_est)
                est_svm2.Est()
                est_svm2.SimulateMEAtMean()
                lin_svm2[i,:,:] = est_svm2.lin
                y_svm2[i,:,:] = est_svm2.y_pred_ME
                for j in range(setting_sim['N_char']):
                    #lr2 = linear_model.LinearRegression(fit_intercept=setting_est['add_const_x'])
                    lr2 = linear_model.LinearRegression(fit_intercept=True)
                    lr2.fit( est_svm2.lin[:,j].reshape([-1,1]),est_svm2.y_pred_ME[:,j] )
                    bhat_svm2[i,j] = lr2.coef_
                
                #plt.plot(est_svm2.lin[:,0], est_svm2.y_pred_ME[:,0])
                '''
            print(setting_sim)
            print(setting_est)
            print('bhat_ols:%s' %np.mean(bhat_ols,axis=0) )    
            print('bhat_2sls:%s' %np.mean(bhat_2sls,axis=0) )
            print('bhat_svm1_linear:%s' %np.mean(bhat_svm1_linear,axis=0) )
            print('bhat_svm1_rbf:%s' %np.mean(bhat_svm1_rbf,axis=0) )
            #print('bhat_svm2:%s' %np.mean(bhat_svm2,axis=0) )
            result = {'setting_sim':setting_sim,'setting_est':setting_est,'bhat_ols':bhat_ols,'bhat_2sls':bhat_2sls,'bhat_svm1_linear':bhat_svm1_linear,'bhat_svm1_rbf':bhat_svm1_rbf,'bhat_svm1_rbf_cv5':bhat_svm1_rbf_cv5,\
                      'bhat_ols_mean':np.mean(bhat_ols,axis=0) ,'bhat_2sls_mean':np.mean(bhat_2sls,axis=0),'bhat_svm1_linear_mean':np.mean(bhat_svm1_linear,axis=0),'bhat_svm1_rbf_mean':np.mean(bhat_svm1_rbf,axis=0),'bhat_svm1_rbf_cv5_mean':np.mean(bhat_svm1_rbf_cv5,axis=0),\
                      'bhat_ols_std':np.std(bhat_ols,axis=0) ,'bhat_2sls_std':np.std(bhat_2sls,axis=0),'bhat_svm1_linear_std':np.std(bhat_svm1_linear,axis=0),'bhat_svm1_rbf_std':np.std(bhat_svm1_rbf,axis=0),'bhat_svm1_rbf_cv5_std':np.std(bhat_svm1_rbf_cv5,axis=0)}
            
            '''
            result = {'setting_sim':setting_sim,'setting_est':setting_est,'bhat_ols':bhat_ols,'bhat_2sls':bhat_2sls,'bhat_svm1_linear':bhat_svm1_linear,'bhat_svm1_rbf':bhat_svm1_rbf,'bhat_svm1_rbf_cv5':bhat_svm1_rbf_cv5,\
                      'bhat_ols_mean':np.mean(bhat_ols,axis=0) ,'bhat_2sls_mean':np.mean(bhat_2sls,axis=0),'bhat_svm1_linear_mean':np.mean(bhat_svm1_linear,axis=0),'bhat_svm1_rbf_mean':np.mean(bhat_svm1_rbf,axis=0),'bhat_svm1_rbf_cv5_mean':np.mean(bhat_svm1_rbf_cv5,axis=0),\
                      'bhat_ols_std':np.std(bhat_ols,axis=0) ,'bhat_2sls_std':np.std(bhat_2sls,axis=0),'bhat_svm1_linear_std':np.std(bhat_svm1_linear,axis=0),'bhat_svm1_rbf_std':np.std(bhat_svm1_rbf,axis=0),'bhat_svm1_rbf_cv5_std':np.std(bhat_svm1_rbf_cv5,axis=0)}
            '''
            results_all.append(result)
    end = time.time()
    time_calc = end-start
    print('Total Time:'+str(time_calc))
    DateCalc=datetime.date.today().strftime('%b-%d-%Y')
    np.save('results_all_'+DateCalc+'.npy',results_all)
    '''
    #Generate table
    OLS=[]
    TSLS=[]
    SVR_LS_linear=[]
    SVR_LS_RBF=[]
    SVR_LS_RBF_cv5=[]
    
    for result in results_all:
        OLS.extend([result['bhat_ols_mean'][0], '('+str(result['bhat_ols_std'][0])+')' ] )
        TSLS.extend([result['bhat_2sls_mean'][0], '('+str( result['bhat_2sls_std'][0] )+')' ] )
        SVR_LS_linear.extend([result['bhat_svm1_linear_mean'][0], '('+str( result['bhat_svm1_linear_std'][0] )+')' ] )
        SVR_LS_RBF.extend([result['bhat_svm1_rbf_mean'][0], '('+str( result['bhat_svm1_rbf_std'][0] )+')' ] )
        SVR_LS_RBF_cv5.extend([result['bhat_svm1_rbf_cv5_mean'][0], '('+str( result['bhat_svm1_rbf_cv5_std'][0] )+')' ] )
    results_dict = {'0OLS':OLS, '2SLS':TSLS, 'SVR-LS (linear)':SVR_LS_linear, 'SVR-LS (rbf), cv=3':SVR_LS_RBF, 'SVR-LS (rbf),cv=5':SVR_LS_RBF_cv5}
    df_index=[]
    for i in range(n_settings):
        df_index.extend( [ 'setting'+str(i),''  ] )
    df = pd.DataFrame( results_dict, index=df_index )
    with open('SVM_results_table_'+DateCalc+'.tex','w') as output:
        output.write(df.to_latex())
    ''' 
    

        
