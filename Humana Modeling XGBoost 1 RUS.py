# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:59:19 2021

@author: adesuyi-m
"""

#AdvancedAnalytics Packages
from AdvancedAnalytics.Forest import forest_classifier
from AdvancedAnalytics.Tree import tree_classifier
from AdvancedAnalytics.Regression import logreg, stepwise
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.NeuralNetwork import nn_classifier
from AdvancedAnalytics.Internet import Metrics

#SKLearn Packages
from sklearn.metrics         import f1_score,precision_score,recall_score,confusion_matrix,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, roc_auc_score, f1_score,precision_score,recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer



#Import Python Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

#Import RUS Package
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

#Import XGBoost Package
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb


folder = "C:\\Users\\adesuyi-m\\OneDrive - Texas A&M University\\Documents\\Humana Competition\\"



csv_read=False
pickle_read=True
xgboost_eval = True

if csv_read:
    file = "Humana Training Set Model-Ready.csv"
    df = pd.read_csv(folder+file)
    df.drop('Unnamed: 0',axis=1,inplace=True)
    X = df.drop('covid_vaccination',axis=1)
    y = df['covid_vaccination']
    
if pickle_read:
    file = "Humana Training Set Model-Ready.pkl"
    df = pd.read_pickle(folder+file)
    X = df.drop('covid_vaccination',axis=1)
    y = df['covid_vaccination']
    y = y.replace({0:1,1:0})

    
if xgboost_eval:
    print('XGBoost with Hyperparameter Optimization')
    start = datetime.now()
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,
                                                     random_state=12345)
    #scaler = MinMaxScaler()
    #X_cols = X.columns.tolist()
    rus = RandomUnderSampler(sampling_strategy={0:118617,1:118617})    
    X_train, y_train = rus.fit_resample(X_train,y_train)
    #smote = SMOTE(random_state=12345)
    #X_train, y_train = smote.fit_resample(X_train,y_train)
    #X_train = pd.DataFrame(scaler.fit_transform(X_train.to_numpy()),columns = X_cols)
    #X_test = pd.DataFrame(scaler.fit_transform(X_test.to_numpy()),columns = X_cols)
    evalset = [(X_train, y_train), (X_test,y_test)]

    final_cl = XGBClassifier(max_depth = 4,learning_rate = 0.3,gamma = 1,
                            reg_lambda = 1,subsample = 0.3,
                            colsample_bytree = 0.7,objective='binary:logistic',
                            eval_metric='auc',n_estimators=5000)
    final_cl = final_cl.fit(X_train,y_train,eval_set=evalset,early_stopping_rounds=15,verbose=True)
    y_train_pred = final_cl.predict(X_train)
    y_test_pred = final_cl.predict(X_test)
    results = final_cl.evals_result()
    
    train_auc = roc_auc_score(y_train,final_cl.predict_proba(X_train)[:,1])
    train_f1 = f1_score(y_train,y_train_pred)
    train_precision = precision_score(y_train,y_train_pred)
    train_recall = recall_score(y_train,y_train_pred)
    train_acc = accuracy_score(y_train,y_train_pred)
    train_misc = 1 - train_acc
    conf_mat_train = confusion_matrix(y_train,y_train_pred)
    
    test_auc = roc_auc_score(y_test,final_cl.predict_proba(X_test)[:,1])
    test_f1 = f1_score(y_test,y_test_pred)
    test_precision = precision_score(y_test,y_test_pred)
    test_recall = recall_score(y_test,y_test_pred)
    test_acc = accuracy_score(y_test,y_test_pred)
    test_misc = 1 - test_acc
    conf_mat_test = confusion_matrix(y_test,y_test_pred)
    print('Training Results')
    print('-'*27)
    print('|{:<15} | {:>7}|'.format('Metric','Value'))
    print("-"*27)
    print('|{:.<15s} | {:>7.4f}|'.format('AUC',train_auc))
    print('|{:.<15s} | {:>7.4f}|'.format('F1',train_f1))
    print('|{:.<15s} | {:>7.4f}|'.format('Precision',train_precision))
    print('|{:.<15s} | {:>7.4f}|'.format('Sensitivity',train_recall))
    print('|{:.<15s} | {:>7.4f}|'.format('Accuracy',train_acc))
    print('|{:.<15s} | {:>7.4f}|'.format('Misc. Rate',train_misc))
    print("-"*27)
    print('-'*27)

    print('-'*67)
    print('Training Confusion Matrix')
    print('-'*67)
    print(' {:<20s} | {:^20s}| {:^20s}|'.format('','Predicted Vacc',
                                     'Predicted No Vacc')) 
    print('-'*67)
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual Vacc',
                                         conf_mat_train[0,0],
                                         conf_mat_train[0,1]))
    print('-'*67)    
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual No Vacc',
                                         conf_mat_train[1,0],
                                         conf_mat_train[1,1]))  
    print('-'*67)
    print('Test Results')
    print('-'*27)
    print('|{:<15} | {:>7}|'.format('Metric','Value'))
    print("-"*27)
    print('|{:.<15s} | {:>7.4f}|'.format('AUC',test_auc))
    print('|{:.<15s} | {:>7.4f}|'.format('F1',test_f1))
    print('|{:.<15s} | {:>7.4f}|'.format('Precision',test_precision))
    print('|{:.<15s} | {:>7.4f}|'.format('Sensitivity',test_recall))
    print('|{:.<15s} | {:>7.4f}|'.format('Accuracy',test_acc))
    print('|{:.<15s} | {:>7.4f}|'.format('Misc. Rate',test_misc))
    print("-"*27)
    print('-'*27)
    print('-'*67)
    print('Test Confusion Matrix')
    print('-'*67)
    print(' {:<20s} | {:^20s}| {:^20s}|'.format('','Predicted Vacc',
                                     'Predicted No Vacc')) 
    print('-'*67)
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual Vacc',
                                         conf_mat_test[0,0],
                                         conf_mat_test[0,1]))
    print('-'*67)    
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual No Vacc',
                                         conf_mat_test[1,0],
                                         conf_mat_test[1,1]))  
    print('-'*67)
    
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0,epochs)
    fig,ax = plt.subplots()
    ax.plot(x_axis,results['validation_0']['auc'],label='Train')
    ax.plot(x_axis,results['validation_1']['auc'],label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('AUC')
    fig.suptitle('XGBoost AUC Over Iterations')
    plt.show()
    
    fig,ax = plt.subplots()
    plot_roc_curve(final_cl,X_train,y_train,ax=ax,name='Training Set')
    plot_roc_curve(final_cl,X_test,y_test,ax=ax,name='Test Set')
        
    fig,ax = plt.subplots()
    fig.suptitle('Training Set Confusion Matrix')
    plot_confusion_matrix(final_cl,X_train,y_train,display_labels=['Vaccinated','Not Vaccinated'],ax=ax) 
    plt.show()
    
    fig,ax = plt.subplots()
    fig.suptitle('Validation Set Confusion Matrix')
    plot_confusion_matrix(final_cl,X_test,y_test,display_labels=['Vaccinated','Not Vaccinated'],ax=ax)
    plt.show()    
    
    xgb.plot_importance(final_cl,height=0.6,max_num_features=20)
    
    
    
    
    