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
from sklearn.metrics import explained_variance_score, mean_squared_error
import pandas as pd

import pickle
import argparse

from thundersvm.python.thundersvm import SVR

parser = argparse.ArgumentParser()
parser.add_argument('--n_train', type=int, default=3000)
parser.add_argument('--dim_inst', type=int, default=5)
parser.add_argument('--dim_x', type=int, default=1)
parser.add_argument('--x_endogenous_idx',  default=[0])
parser.add_argument('--sd_inst',  default=1.)
parser.add_argument('--var_u',  default=1., help='Variance of noise in 1st stage.')
parser.add_argument('--var_eps',  default=1., help='Variance of epsilon in 2nd stage')
parser.add_argument('--cov_u_eps',  default=.2, help='Covariance between epsion and u of endogenous x. Strength of endogeneity.')
parser.add_argument('--func_1st',  default=np.sinc, help='Nonlinear 1st stage function of DGP.')

z_polynomial = 2
n_rep=50

df = pd.DataFrame(columns= ['func','func_params','n_train','dim_inst','var_noise_1st', 'var_noise_2nd', 'cov_noises',\
     'bias_lr', 'bias_2sls', 'bias_quad2sls', 'bias_svm', 'mad_lr', 'mad_2sls', 'mad_quad2sls', 'mad_svm'\
         'var_lr', 'var_2sls', 'var_quad2sls', 'var_svm',\
              'r2_1st_2sls', 'r2_1st_quad2sls', 'r2_1st_svm',\
                  'valmse_1st_2sls', 'valmse_1st_quad2sls', 'valmse_1st_svm']  )

n_settings = 27
settings = []
for i in range(n_settings):
    a = parser.parse_args()
    if i%3==0: a.var_u = 1. 
    if i%3==1: a.var_u = 4. 
    if i%3==2: a.var_u = 16. 

    if int(i/3)%3==0: a.var_eps = 1. 
    if int(i/3)%3==1: a.var_eps = 4. 
    if int(i/3)%3==2: a.var_eps = 16. 

    if int(i/9)%3==0: a.cov_u_eps = np.sqrt( a.var_u*a.var_eps )*.25 
    if int(i/9)%3==1: a.cov_u_eps = np.sqrt( a.var_u*a.var_eps )*.50 
    if int(i/9)%3==2: a.cov_u_eps = np.sqrt( a.var_u*a.var_eps )*.75 

    settings.append(a)



for setting_id in range(n_settings ):
    args = settings[setting_id]
    args.x_exogenous_idx = [item for item in list(range(args.dim_x)) if item not in args.x_endogenous_idx]
    args.n_val = int( args.n_train*.5 )
    args.n = args.n_train + args.n_val

    args.x_exogenous_idx = [item for item in list(range(args.dim_x)) if item not in args.x_endogenous_idx]
    args.n_val = int( args.n_train*.5 )
    args.n = args.n_train + args.n_val

    df.loc[setting_id,'func'] = args.func_1st.__name__
    df.loc[setting_id,'n_train'] = args.n_train
    df.loc[setting_id,'dim_inst'] = args.dim_inst
    df.loc[setting_id,'var_noise_1st'] = args.var_u
    df.loc[setting_id,'var_noise_2nd'] = args.var_eps
    df.loc[setting_id,'cov_noises'] = args.cov_u_eps

    result_bias = np.zeros([n_rep, 4])
    result_r2 = np.zeros([n_rep, 3])
    result_valerror = np.zeros([n_rep, 3])
    for it in range(n_rep):
        print('---------setting {}, rep {}----------'.format(setting_id, it))
        t = time.time()

        #---------Simulate data-------------
        ##Covariance matrix of x and epsilon. The first element is epsilon, remaining dim_x elements are x.
        cov_error = np.identity(args.dim_x+1)* args.var_u 
        cov_error[0,0]  = args.var_eps 
        cov_error[0,np.array(args.x_endogenous_idx)+1 ] = args.cov_u_eps
        cov_error[np.array(args.x_endogenous_idx)+1,0] = args.cov_u_eps

        ##The error term fo the fist and second stage.
        ##The first stage error term u is correlated with eps, only for x_endogenous
        mean_error = np.zeros(cov_error.shape[0])
        errors = np.random.multivariate_normal(mean=mean_error,cov=cov_error,size=args.n)
        eps = errors[:,0] #Error term of the second stage.
        u = errors[:,1:] #Error term of the first stage.

        ##Generate instrumental varables.
        inst_data = np.random.uniform(-3,3,size=[args.n,args.dim_inst])#np.random.normal(size=[n,dim_inst]) 
        genpoly = preprocessing.PolynomialFeatures(degree=z_polynomial)

        ##Construct x from z.
        ##Endogenous x are function of z plus u.
        ##Exogenous x are diretly u.
        x_data = np.zeros( [args.n, args.dim_x] )
        x_data[:, args.x_exogenous_idx] = u[:,args.x_exogenous_idx]
        ##z is exogenous instrument (inst_data) + inclusive IV (x_data[:,x_exogenous_idx])
        z_data = np.concatenate( (inst_data ,x_data[:,args.x_exogenous_idx]),axis=1) 
        for j in args.x_endogenous_idx:
            z_poly =  - np.power(z_data, 2.)#2. #-genpoly.fit_transform(z_data)#
            #x_data[:,j] =  np.matmul( np.exp(z_poly ) , np.random.randn( z_poly.shape[1] ).reshape([-1,1]) *.5 ).flatten() + u[:,j]
            #x_data[:,j] = np.sum(  np.exp(z_poly ), axis=1)  + u[:,j]
            x_data[:,j] = np.sum(  args.func_1st(z_poly ), axis=1)  + u[:,j]


        ##y is a fucntion of x.
        y_data = np.sum(x_data,axis=1).flatten() + eps



        #------------Endogenous linear regression.--------------
        lr1=linear_model.LinearRegression()
        lr1.fit(x_data[:args.n_train,:],y_data[:args.n_train])
        bhat_lr1 = lr1.coef_
        print('Estimate of linear reg: {}'.format(bhat_lr1[args.x_endogenous_idx]) )
        result_bias[it,0] = bhat_lr1-1.





        #------------Traditional IV regression.--------------
        ##1st stage
        x_hat = np.zeros_like(x_data[:args.n_train,:])
        x_hat_val = np.zeros_like(x_data[args.n_train:,:])
        for j in args.x_endogenous_idx:
            lr_1st = linear_model.LinearRegression()
            lr_1st.fit(z_data[:args.n_train,:],x_data[:args.n_train,j])
            x_hat[:,j]=lr_1st.predict(z_data[:args.n_train,:])
            x_hat_val[:,j]=lr_1st.predict(z_data[args.n_train:,:])
        x_hat[:, args.x_exogenous_idx] = x_data[:args.n_train, args.x_exogenous_idx]
        evs_2sls = explained_variance_score(x_data[:args.n_train,args.x_endogenous_idx], x_hat[:,args.x_endogenous_idx]  )
        val_error = mean_squared_error(x_data[args.n_train:,:],x_hat_val)
        ##2nd stage
        lr_2nd=linear_model.LinearRegression()
        lr_2nd.fit(x_hat,y_data[:args.n_train])
        bhat_iv = lr_2nd.coef_
        print('Estimate of 2SLS: {}'.format(bhat_iv[args.x_endogenous_idx]) , '(evs: {:.4f}, val_mse:{:.4f})'.format(evs_2sls,val_error) )
        result_bias[it,1] = bhat_iv-1.
        result_r2[it,0] = evs_2sls
        result_valerror[it,0] = val_error



        #------------Quadratic IV regression.--------------
        ##1st stage
        x_hat = np.zeros_like(x_data[:args.n_train,:])
        x_hat[:,j]=lr_1st.predict(z_data[:args.n_train,:])
        for j in args.x_endogenous_idx:
            lr_1st = linear_model.LinearRegression()
            genpoly = preprocessing.PolynomialFeatures(degree=2)
            z_poly = genpoly.fit_transform(z_data[:args.n_train,:])
            z_poly_val = genpoly.fit_transform(z_data[args.n_train:,:])
            lr_1st.fit(z_poly,x_data[:args.n_train,j])
            x_hat[:,j]=lr_1st.predict(z_poly)
            x_hat_val[:,j]=lr_1st.predict(z_poly_val)
        x_hat[:, args.x_exogenous_idx] = x_data[:args.n_train, args.x_exogenous_idx]
        evs_2sls_quad = explained_variance_score(x_data[:args.n_train,args.x_endogenous_idx], x_hat[:,args.x_endogenous_idx]  )
        val_error = mean_squared_error(x_data[args.n_train:,:],x_hat_val)
        ##2nd stage
        lr_2nd=linear_model.LinearRegression()
        lr_2nd.fit(x_hat,y_data[:args.n_train])
        bhat_iv_quad = lr_2nd.coef_
        print('Estimate of 2SLS (quad): {}'.format(bhat_iv_quad[args.x_endogenous_idx]) , '(evs: {:.4f}, val_mse:{:.4f})'.format(evs_2sls_quad,val_error))
        result_bias[it,2] = bhat_iv_quad-1.
        result_r2[it,1] = evs_2sls
        result_valerror[it,1] = val_error




        #------------SVM-IV regression.--------------
        ##1st stage
        #param_grid = [{'C': [.01,1.,100.], 'gamma': [.01,1.,100.], 'kernel': [ 'rbf' ]}]
        ##Smlaller C for simpler function
        ##Low gamma for further influence
        param_grid = [{'C': [1.,10.,100.], 'gamma': [1/args.dim_inst], 'kernel': [ 'rbf' ]}]
        n_cv=2
        x_hat = np.zeros_like(x_data[:args.n_train,:])
        x_hat_val = np.zeros_like(x_data[args.n_train:,:])
        for j in args.x_endogenous_idx:
            #svr_rbf = svm.SVR()
            svr_rbf = SVR()
            gridsearch = model_selection.GridSearchCV(svr_rbf,param_grid=param_grid,refit=True, cv=n_cv)
            gridsearch.fit(z_data[:args.n_train,:],x_data[:args.n_train,j])
            x_hat[:,j]=gridsearch.predict(z_data[:args.n_train,:])
            x_hat_val[:,j]=gridsearch.predict(z_data[args.n_train:,:])
        x_hat[:, args.x_exogenous_idx] = x_data[:args.n_train, args.x_exogenous_idx]
        evs_svm = explained_variance_score(x_data[:args.n_train,args.x_endogenous_idx], x_hat[:,args.x_endogenous_idx]  )
        val_error = mean_squared_error(x_data[args.n_train:,:],x_hat_val)

        ##2nd stage
        lr_2nd=linear_model.LinearRegression()
        lr_2nd.fit(x_hat,y_data[:args.n_train])
        bhat_svm1 = lr_2nd.coef_
        print('Estimate of SVR-IV: {}'.format(bhat_svm1[args.x_endogenous_idx]), '(evs: {:.4f}, val_mse:{:.4f})'.format(evs_svm,val_error) )
        result_bias[it,3] = bhat_svm1-1.
        result_r2[it,2] = evs_svm
        result_valerror[it,2] = val_error

        print('(time: {:.3f})'.format(time.time()-t))



    df.loc[setting_id,'bias_lr'] = result_bias.mean(axis=0)[0]
    df.loc[setting_id,'bias_2sls'] = result_bias.mean(axis=0)[1]
    df.loc[setting_id,'bias_quad2sls'] = result_bias.mean(axis=0)[2]
    df.loc[setting_id,'bias_svm'] = result_bias.mean(axis=0)[3]

    df.loc[setting_id,'mad_lr'] = np.abs(result_bias).mean(axis=0)[0]
    df.loc[setting_id,'mad_2sls'] = np.abs(result_bias).mean(axis=0)[1]
    df.loc[setting_id,'mad_quad2sls'] = np.abs(result_bias).mean(axis=0)[2]
    df.loc[setting_id,'mad_svm'] = np.abs(result_bias).mean(axis=0)[3]

    df.loc[setting_id,'var_lr'] = result_bias.var(axis=0)[0]
    df.loc[setting_id,'var_2sls'] = result_bias.var(axis=0)[1]
    df.loc[setting_id,'var_quad2sls'] = result_bias.var(axis=0)[2]
    df.loc[setting_id,'var_svm'] = result_bias.var(axis=0)[3]

    df.loc[setting_id,'r2_2sls'] = result_r2.mean(axis=0)[0]
    df.loc[setting_id,'r2_quad2sls'] = result_r2.mean(axis=0)[1]
    df.loc[setting_id,'r2_svm'] = result_r2.mean(axis=0)[2]

    df.loc[setting_id,'valmse_2sls'] = result_valerror.mean(axis=0)[0]
    df.loc[setting_id,'valmse_quad2sls'] = result_valerror.mean(axis=0)[1]
    df.loc[setting_id,'valmse_svm'] = result_valerror.mean(axis=0)[2]


df_all = pickle.load( open('svm_result_df.pickle','rb')  )
df_all = pd.concat( [df_all, df] )
pickle.dump(df_all, open('svm_result_df.pickle','wb'))
