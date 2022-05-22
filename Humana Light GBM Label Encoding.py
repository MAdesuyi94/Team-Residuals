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

cats = ['rx_overall_mbr_resp_pmpm_cost_t_',
         'rx_overall_dist_gpi6_pmpm_ct_t_6',
         'rx_maint_net_paid_pmpm_cost_t_12',
         'rej_med_ip_snf_coins_pmpm_cost_t',
         'med_physician_office_allowed_pmp',
         'hum_region',
         'rx_nonmail_dist_gpi6_pmpm_ct_t_9',
         'rej_med_er_net_paid_pmpm_cost_t_',
         'med_outpatient_mbr_resp_pmpm_cos',
         'total_ip_maternity_net_paid_pmpm',
         'total_physician_office_mbr_resp_',
         'med_outpatient_visit_ct_pmpm_t_1',
         'total_med_net_paid_pmpm_cost_t_6',
         'rx_nonmaint_dist_gpi6_pmpm_ct_t_',
         'rx_generic_dist_gpi6_pmpm_ct_t_9',
         'total_bh_copay_pmpm_cost_t_9_6_3',
         'rx_maint_pmpm_cost_t_12_9_6m_b4',
         'rx_nonbh_pmpm_cost_t_9_6_3m_b4',
         'rx_generic_pmpm_cost_t_6_3_0m_b4',
         'rx_phar_cat_humana_pmpm_ct_t_9_6',
         'rx_overall_gpi_pmpm_ct_t_6_3_0m_',
         'mcc_chf_pmpm_ct_t_9_6_3m_b4',
         'rx_maint_pmpm_cost_t_6_3_0m_b4',
         'rx_branded_pmpm_ct_t_6_3_0m_b4',
         'total_allowed_pmpm_cost_t_9_6_3m',
         'oontwk_mbr_resp_pmpm_cost_t_6_3_',
         'rx_nonbh_net_paid_pmpm_cost_t_6_',
         'rx_gpi2_39_pmpm_cost_t_6_3_0m_b4',
         'rx_maint_pmpm_ct_t_6_3_0m_b4',
         'rx_mail_net_paid_pmpm_cost_t_6_3',
         'rx_mail_mbr_resp_pmpm_cost_t_9_6',
         'rx_nonbh_pmpm_ct_t_9_6_3m_b4',
         'rx_gpi2_62_pmpm_cost_t_9_6_3m_b4',
         'rx_overall_gpi_pmpm_ct_t_12_9_6m',
         'rx_nonotc_pmpm_cost_t_6_3_0m_b4',
         'rx_maint_net_paid_pmpm_cost_t_9_',
         'rx_phar_cat_cvs_pmpm_ct_t_9_6_3m',
         'REP_bh_urgent_care_copay_pmpm_co',
         'REP_cons_hhcomp',
         'REP_cons_mobplus',
         'REP_mcc_ano_pmpm_ct_t_9_6_3m_b4',
         'REP_med_ambulance_coins_pmpm_cos',
         'REP_med_outpatient_deduct_pmpm_c',
         'REP_total_physician_office_visit',
         'IMP_rej_med_outpatient_visit_ct_',
         'IMP_rx_gpi2_17_pmpm_cost_t_12_9_','bh_ncdm_ind','IMP_atlas_retirement_destination',
        'IMP_atlas_hiamenity','IMP_atlas_hipov_1115',
        'IMP_atlas_type_2015_mining_no','IMP_atlas_low_employment_2015_up',
        'bh_ncal_ind','IMP_atlas_type_2015_recreation_n',
        'IMP_atlas_population_loss_2015_u','IMP_atlas_farm_to_school13',
        'IMP_atlas_persistentchildpoverty','IMP_atlas_perpov_1980_0711',
        'IMP_atlas_low_education_2015_upd','sex_cd']



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
    file = 'ImputedData_SAS_EM_Train_Label_Encoded.pkl'
    df = pd.read_pickle(folder+file)
    df = df.fillna(0)
    X = df.drop('covid_vaccination',axis=1)
    y = df['covid_vaccination']
    cat_list = [df.columns.get_loc(c) for c in cats if c in df]
if lightgbm_eval:
    print('LightGBM with Hyperparameter Optimization')
    start = datetime.now()
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y)
    #smote = SMOTENC(random_state=12345,categorical_features=df[cats])
    #X_train, y_train = smote.fit_resample(X_train,y_train)
#    scaler = MinMaxScaler()
#    X_cols = X_train.columns.tolist()
#    X_train = pd.DataFrame(scaler.fit_transform(X_train.to_numpy()),columns = X_cols)
#    X_test = pd.DataFrame(scaler.fit_transform(X_test.to_numpy()),columns = X_cols)

  # ,class_weight={0:1,1:5} 
    lgb_train = lgb.Dataset(X_train,label=y_train,categorical_feature=cats)
    lgb_test = lgb.Dataset(X_test,label=y_test)
    parameters = {'objective':'binary',
                  'metric':'auc',
                  'is_unbalance':'true',
                  'boosting':'gbdt',
                  'num_leaves':31,
                  'feature_fraction':0.5,
                  'bagging_fraction':0.5,
                  'learning_rate':0.01,
                  'verbose':1
                  }
    evalset = {}
    model = lgb.train(params=parameters,train_set=lgb_train,num_boost_round=5000,valid_sets=lgb_test,early_stopping_rounds=50,evals_result=evalset)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    results = model.evals_result_
    
    train_auc = roc_auc_score(y_train,y_train_pred)
    train_f1 = f1_score(y_train,y_train_pred)
    train_precision = precision_score(y_train,y_train_pred)
    train_recall = recall_score(y_train,y_train_pred)
    train_acc = accuracy_score(y_train,y_train_pred)
    train_misc = 1 - train_acc
    conf_mat_train = confusion_matrix(y_train,y_train_pred)
    
    test_auc = roc_auc_score(y_test,y_test_pred)
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
    print(' {:<20s} | {:^20s}| {:^20s}|'.format('','Predicted No Vacc',
                                     'Predicted Vacc')) 
    print('-'*67)
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual No Vacc',
                                         conf_mat_train[0,0],
                                         conf_mat_train[0,1]))
    print('-'*67)    
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual Vacc',
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
    print(' {:<20s} | {:^20s}| {:^20s}|'.format('','Predicted No Vacc',
                                     'Predicted Vacc')) 
    print('-'*67)
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual No Vacc',
                                         conf_mat_test[0,0],
                                         conf_mat_test[0,1]))
    print('-'*67)    
    print('|{:.<20s} | {:^20d}| {:^20d}|'.format('Actual Vacc',
                                         conf_mat_test[1,0],
                                         conf_mat_test[1,1]))  
    print('-'*67)
    
#    epochs = len(results['training']['auc'])
#    x_axis = range(0,epochs)
#    fig,ax = plt.subplots()
#    ax.plot(x_axis,results['training']['auc'],label='Train')
#    ax.plot(x_axis,results['valid_1']['auc'],label='Test')
#    ax.set_xlabel('Epochs')
#    ax.set_ylabel('AUC')
#    fig.suptitle('LightGBM AUC Over Iterations')
#    plt.show()
#    
#    fig,ax = plt.subplots()
#    ax.plot(x_axis,results['training']['binary_logloss'],label='Train')
#    ax.plot(x_axis,results['valid_1']['binary_logloss'],label='Test')
#    ax.set_xlabel('Epochs')
#    ax.set_ylabel('Log Loss')
#    fig.suptitle('LightGBM Log Loss Over Iterations')
#    plt.show()