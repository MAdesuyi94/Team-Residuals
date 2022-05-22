# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:14:27 2021

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
from imblearn.over_sampling import SMOTE, SMOTENC

#Import XGBoost Package
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb


folder = "C:\\Users\\adesuyi-m\\Documents\\Humana Competition\\"

cat_list = ['src_div_id','total_bh_copay_pmpm_cost_t_9-6-3m_b4',
               'mcc_ano_pmpm_ct_t_9-6-3m_b4','rx_maint_pmpm_cost_t_12-9-6m_b4',
               'rx_nonbh_pmpm_cost_t_9-6-3m_b4','rx_gpi2_17_pmpm_cost_t_12-9-6m_b4','rx_generic_pmpm_cost_t_6-3-0m_b4',
               'rx_overall_mbr_resp_pmpm_cost_t_6-3-0m_b4','rx_overall_dist_gpi6_pmpm_ct_t_6-3-0m_b4',
               'rx_phar_cat_humana_pmpm_ct_t_9-6-3m_b4',
               'rx_overall_gpi_pmpm_ct_t_6-3-0m_b4','mcc_chf_pmpm_ct_t_9-6-3m_b4',
               'bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4','rx_maint_pmpm_cost_t_6-3-0m_b4',
               'cons_mobplus','rx_maint_net_paid_pmpm_cost_t_12-9-6m_b4',
               'rej_med_outpatient_visit_ct_pmpm_t_6-3-0m_b4',
               'rej_med_ip_snf_coins_pmpm_cost_t_9-6-3m_b4','med_physician_office_allowed_pmpm_cost_t_9-6-3m_b4',
               'total_physician_office_net_paid_pmpm_cost_t_9-6-3m_b4',
               'rx_branded_pmpm_ct_t_6-3-0m_b4','med_outpatient_deduct_pmpm_cost_t_9-6-3m_b4',
               'total_allowed_pmpm_cost_t_9-6-3m_b4',
               'cms_orig_reas_entitle_cd','oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4',
               'hum_region','rx_nonmail_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
               'rej_med_er_net_paid_pmpm_cost_t_9-6-3m_b4','med_outpatient_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'rx_nonbh_net_paid_pmpm_cost_t_6-3-0m_b4',
               'rx_gpi2_39_pmpm_cost_t_6-3-0m_b4','atlas_type_2015_update',
               'total_ip_maternity_net_paid_pmpm_cost_t_12-9-6m_b4',
               'rx_maint_pmpm_ct_t_6-3-0m_b4','rx_mail_net_paid_pmpm_cost_t_6-3-0m_b4',
               'total_physician_office_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'rx_mail_mbr_resp_pmpm_cost_t_9-6-3m_b4',
               'med_outpatient_visit_ct_pmpm_t_12-9-6m_b4','rx_nonbh_pmpm_ct_t_9-6-3m_b4',
               'total_med_net_paid_pmpm_cost_t_6-3-0m_b4','rx_gpi2_62_pmpm_cost_t_9-6-3m_b4',
               'rx_overall_gpi_pmpm_ct_t_12-9-6m_b4','cons_hhcomp',
               'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4','rx_nonotc_pmpm_cost_t_6-3-0m_b4',
               'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4',
               'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4','bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4',
               'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4','total_physician_office_visit_ct_pmpm_t_6-3-0m_b4',
               'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4','race_cd',
               'bh_ncdm_ind','atlas_retirement_destination_2015_upda','atlas_hiamenity',
               'atlas_hipov_1115','atlas_type_2015_mining_no',
               'atlas_low_employment_2015_update','bh_ncal_ind','atlas_type_2015_recreation_no',
               'atlas_population_loss_2015_update','atlas_farm_to_school13','sex_cd',
               'atlas_persistentchildpoverty_1980_2011','atlas_perpov_1980_0711'
               ,'atlas_low_education_2015_update']



csv_read=False
pickle_read=True
lightgbm_eval = True

if csv_read:
    file = "ImputedData_Train.csv"
    df = pd.read_csv(folder+file)
   # df.drop('Unnamed: 0',axis=1,inplace=True)
    X = df.drop('covid_vaccination',axis=1)
    y = df['covid_vaccination']

    
    xcols = X.columns.tolist()
    
    dt = X.dtypes
    dt = dt.reset_index()
    cols = dt.loc[dt[0] == 'object','index'].tolist()    
      
    
if pickle_read:
    file = 'Humana Training Data Label Encoded.pkl'
    df = pd.read_pickle(folder+file)
    #df['covid_vaccination'] = df['covid_vaccination'].replace({0:1,1:0})
    X = df.drop('covid_vaccination',axis=1)
    y = df['covid_vaccination']
    cat_num_list = [X.columns.get_loc(i) for i in X.columns.tolist() if i in cat_list]
    
if lightgbm_eval:
    print('LightGBM with Hyperparameter Optimization')
    start = datetime.now()    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=12345)
    evalset = [(X_train, y_train), (X_test,y_test)]
    model = LGBMClassifier(max_depth=-1,num_leaves=31, learning_rate=0.01,n_estimators=50000,
                           n_jobs=-1,bagging_fraction=0.3,colsample_bytree=0.7,
                           early_stopping_round=15,first_metric_only=True,random_state=12345,class_weight={1:1,0:5})
    model = model.fit(X_train,y_train,eval_set=evalset,eval_metric='auc',categorical_feature=cat_list)
    y_train_pred = model.predict(X_train,num_iteration=model.best_iteration_)
    y_test_pred = model.predict(X_test,num_iteration=model.best_iteration_)
    results = model.evals_result_
    
    train_auc = roc_auc_score(y_train,model.predict_proba(X_train,num_iteration=model.best_iteration_)[:,1])
    train_f1 = f1_score(y_train,y_train_pred)
    train_precision = precision_score(y_train,y_train_pred)
    train_recall = recall_score(y_train,y_train_pred)
    train_acc = accuracy_score(y_train,y_train_pred)
    train_misc = 1 - train_acc
    conf_mat_train = confusion_matrix(y_train,y_train_pred)
    
    test_auc = roc_auc_score(y_test,model.predict_proba(X_test,num_iteration=model.best_iteration_)[:,1])
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
    plot_confusion_matrix(model,X_train,y_train,display_labels=['Vaccinated','Not Vaccinated'])

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
    plot_confusion_matrix(model,X_test,y_test,display_labels=['Vaccinated','Not Vaccinated'])
    epochs = len(results['training']['auc'])
    x_axis = range(0,epochs)
    fig,ax = plt.subplots()
    ax.plot(x_axis,results['training']['auc'],label='Train')
    ax.plot(x_axis,results['valid_1']['auc'],label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('AUC')
    fig.suptitle('LightGBM AUC Over Iterations')
    plt.show()
    
    fig,ax = plt.subplots()
    ax.plot(x_axis,results['training']['binary_logloss'],label='Train')
    ax.plot(x_axis,results['valid_1']['binary_logloss'],label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log Loss')
    fig.suptitle('LightGBM Log Loss Over Iterations')
    plt.show()
    lgb.plot_importance(model,max_num_features=20)
    fig,ax = plt.subplots()
    plot_roc_curve(model,X_train,y_train,ax=ax,name='Training Set')
    plot_roc_curve(model,X_test,y_test,ax=ax,name='Test Set')
    plt.show()
    df_holdout = pd.read_pickle(folder+'Humana Holdout Data Label Encoded.pkl')
    df_holdout['Prob'] = model.predict_proba(df_holdout.drop('ID',axis=1),num_iteration=model.best_iteration_)[:,1]
    holdout_export = df_holdout[['ID','Prob']]
    holdout_export['RANK'] = holdout_export['Prob'].rank(ascending=False)
    holdout_export.rename(columns={'Prob':'SCORE'},inplace=True)
    holdout_export.to_csv(folder+'Brandon_Maddox_20211007.csv')
