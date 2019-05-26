from time import time
import numpy as np
from scipy.stats import uniform,randint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import make_pipeline



class Tuning:
    def __init__(self,train_df,response_name,folds,modelList,
                 lossDict):
        '''
        init method for salary linear estimation
        train_df      : preprocessed training set, DataFrame
        response_name : response name in train set, str
        folds         : folds in cv, int
        modelList     : estimator container, list
        lossDict      : estimator key:mse value container, dict 
        '''
        self.train_df = train_df
#         self.Xtest_df = Xtest_df
        # split X features and reponse from train df
        self.Xtrain_df = self.train_df.drop(response_name,axis=1)
        self.folds = folds
        self.ytrain_df = self.train_df[response_name]
        self.modelList = modelList
        self.lossDict = lossDict
        self.lossDict = lossDict

    def plot_loss(self,hyper_arr,train_loss_arr,val_loss_arr,
                  figsz,xlabel,ylabel,folds):
        '''
        visualize loss in loss_arr at each hyperparameter in hyper_arr
        hyper_arr      : hyperparameter value container, list
        train_loss_arr : training loss container, list
        val_loss_arr   : validation loss container, list
        figsize        : dimensions of pyplot figure, tuple
        xlabel         : pyplot x-axis label, str
        ylabel         : pyplot y-axis label, str
        folds          : cv folds in title, int  
        '''
        f,ax = plt.subplots(figsize=figsz)
        ax.plot(hyper_arr,train_loss_arr,color='blue',
                label='train')
        ax.plot(hyper_arr,val_loss_arr,color='red',
                label='validation')
        ax.set(title='{}-Fold Cross-Validation Loss'.format(folds),
               xlabel=xlabel,ylabel=ylabel)
        ax.legend()
        

    def _tune_hyperparams(self,model,params,lossfunc,figsz,ylabel,
                          plot_param,searchRandom=False,
                          n_iter=100,verbose=False):
        '''
        method to tune model hyperparameters with GridSearchCV
        model        : sklearn estimator object, sklearn estimator
        params       : hyperparameter container, dict
        lossfunc     : estimator loss function, str
        figsize      : pyplot figure size, tuple
        ylabel       : name of loss in pyplot, str
        plot_param   : sklearn estimator hyperparameter xlabel, str
        searchRandom : specify whether to execute RandomizedSearchCV, bool
        n_iter       : number of iterations of RandomizedSearchCV to execute, int
        verbose      : controls whether to print train, validation scores, bool
        returns      : search best_parameters
        '''
        if searchRandom:
            rs = RandomizedSearchCV(model,param_distributions=params,
                                    cv=self.folds,scoring=lossfunc,
                                    return_train_score=True,n_iter=n_iter)
            rs.fit(self.Xtrain_df,self.ytrain_df)
            best_model = rs.best_estimator_
            best_score = rs.best_score_
            best_test_score_array = rs.cv_results_['mean_test_score']
            if lossfunc=='neg_mean_squared_error':
                best_score = -1.0*best_score
                best_test_score_array = -1.0*best_test_score_array
                scores_train = -1.0*rs.cv_results_['mean_train_score']
                scores_val = -1.0*rs.cv_results_['mean_test_score']
            else:
                best_score = best_score
                best_test_score_array = best_test_score_array
                scores_train = rs.cv_results_['mean_train_score']
                scores_val = rs.cv_results_['mean_test_score']
            best_params = rs.best_params_    
        else:    
            gs = GridSearchCV(model,param_grid=params,cv=self.folds,
                             scoring=lossfunc,return_train_score=True)
            gs.fit(self.Xtrain_df,self.ytrain_df)
            best_model = gs.best_estimator_
            best_score = gs.best_score_
            best_test_score_array = gs.cv_results_['mean_test_score']
            if lossfunc=='neg_mean_squared_error':
                best_score = -1.0*best_score
                best_test_score_array = -1.0*best_test_score_array
                scores_train = -1.0*gs.cv_results_['mean_train_score']
                scores_val = -1.0*gs.cv_results_['mean_test_score']
            else:
                best_score = best_score
                scores_train = gs.cv_results_['mean_train_score']
                scores_val = gs.cv_results_['mean_test_score']
            best_params = gs.best_params_
        if verbose:    
            print('scores_train:',scores_train)
            print('scores_val:',scores_val)
        print('tuned optimal hyperparmeter:')
        for k in best_params:
            print('{}: {}'.format(k,best_params[k]))
        print('\ngridsearch best score:',best_score)
        # viz loss vs. hyperparameters
        try:
            xs = params[plot_param]
            self.plot_loss(xs,scores_train,scores_val,figsz,
                           plot_param,ylabel,self.folds)
        except: pass     
        # return hyperparameter search best model:    
        return best_model     