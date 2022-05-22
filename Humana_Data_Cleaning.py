# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:12:00 2021

@author: adesuyi-m
"""

import numpy as np
import pandas as pd

folder = 'C:\\Users\\adesuyi-m\\Documents\\Humana Competition\\'
train_file = '2021_Competition_Training.csv'
test_file = '2021_Competition_Holdout.csv'

first_load = True
subsequent_load = True

if first_load:
    train = pd.read_csv(folder+train_file)
    test = pd.read_csv(folder+test_file)
    
    
    train.to_pickle(folder+'2021_Competition_Training.pkl')
    test.to_pickle(folder+'2021_Competition_Test.pkl')
    
if subsequent_load:    
    train = pd.read_pickle(folder+'2021_Competition_Training.pkl')
    test = pd.read_pickle(folder+'2021_Competition_Test.pkl')
    train.drop('Unnamed: 0',axis=1,inplace=True)
    test.drop('Unnamed: 0',axis=1,inplace=True)

dirty_train_columns = [2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,68,75,85,102,124,127,131,132,135,160,174,180,187,192,202,209,210,211,215,220,230,234,240,243,247,251,255,261,285,293,297,300,305,306,309,323,334,344,345,352,353,355,359]
dirty_test_columns = [2,8,9,11,13,16,20,26,28,29,31,33,51,55,58,62,64,66,75,82,85,102,124,131,132,135,159,173,179,191,208,209,210,219,233,239,246,254,260,284,287,292,296,304,305,307,308,322,333,343,344,349,351,352,354]

#brandon_list = list(range(1,92))
#josh_list = list(range(92,184))
matt_list = list(range(184,276))
#sara_list = list(range(276,368))

matt_list.insert(0,0)

dirty_test = []

for i in matt_list:
    if i in dirty_test_columns:
        dirty_test.append(i)

#train.iloc[:,184:276].to_pickle(folder+'2021_Competition_Training_col_184_275_Matt.pkl')
#brandon_clean = train.iloc[:,brandon_list]
#josh_clean = train.iloc[:,josh_list]
matt_clean = train.iloc[:,matt_list]
#sara_clean = train.iloc[:,sara_list]

matt_test_clean = test.iloc[:,matt_list]

test.iloc[:,dirty_test].columns



brandon_dirty_train_columns = ['auth_3mth_post_acute_dia', 'bh_ip_snf_net_paid_pmpm_cost_9to12m_b4',
       'auth_3mth_acute_ckd', 'src_div_id',
       'bh_ip_snf_net_paid_pmpm_cost_3to6m_b4', 'auth_3mth_post_acute_trm',
       'rx_gpi4_6110_pmpm_ct', 'auth_3mth_acute_vco', 'rx_bh_pmpm_ct_0to3m_b4',
       'auth_3mth_dc_ltac', 'auth_3mth_post_acute_inj', 'auth_3mth_dc_home',
       'bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4', 'auth_3mth_dc_no_ref',
       'auth_3mth_dc_snf', 'bh_ip_snf_net_paid_pmpm_cost_0to3m_b4',
       'auth_3mth_psychic', 'auth_3mth_bh_acute', 'auth_3mth_acute_chf',
       'auth_3mth_acute_bld', 'rx_gpi2_34_dist_gpi6_pmpm_ct']

josh_dirty_train_columns = ['lab_albumin_loinc_pmpm_ct', 'rx_gpi2_72_pmpm_ct_6to9m_b4',
       'auth_3mth_acute_res', 'auth_3mth_acute_dig',
       'auth_3mth_dc_acute_rehab', 'bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4',
       'auth_3mth_non_er', 'bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4',
       'auth_3mth_post_acute_cer']

matt_dirty_train_columns = ['auth_3mth_post_acute_mus', 'bh_ip_snf_net_paid_pmpm_cost_6to9m_b4',
       'auth_3mth_post_acute_sns', 'auth_3mth_acute_can',
       'auth_3mth_post_acute', 'auth_3mth_facility',
       'auth_3mth_post_acute_men', 'auth_3mth_home', 'auth_3mth_transplant',
       'rev_cms_ansth_pmpm_ct', 'auth_3mth_acute', 'auth_3mth_dc_left_ama',
       'auth_3mth_acute_ccs_227', 'auth_3mth_dc_custodial',
       'rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4', 'auth_3mth_ltac']

sara_dirty_train_columns = ['auth_3mth_snf_post_hsp', 'auth_3mth_acute_trm',
       'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
       'auth_3mth_snf_direct', 'auth_3mth_dc_home_health',
       'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4', 'auth_3mth_acute_ner',
       'ccsp_065_pmpm_ct', 'auth_3mth_post_er', 'rx_gpi2_33_pmpm_ct_0to3m_b4',
       'auth_3mth_post_acute_chf', 'auth_3mth_dc_other',
       'auth_3mth_bh_acute_mean_los', 'auth_3mth_post_acute_gus',
       'auth_3mth_acute_mus']

matt_dirty_test_columns = ['bh_ip_snf_net_paid_pmpm_cost_6to9m_b4', 'auth_3mth_acute_can',
       'auth_3mth_post_acute', 'auth_3mth_facility', 'auth_3mth_home',
       'rev_cms_ansth_pmpm_ct', 'auth_3mth_acute', 'auth_3mth_acute_ccs_227',
       'rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4', 'auth_3mth_ltac']

for c in matt_dirty_test_columns:
    print(matt_test_clean.loc[:,c].unique())
    
for c in matt_dirty_test_columns:
    matt_test_clean.loc[:,c] = matt_test_clean.loc[:,c].replace('*',np.nan)
    matt_test_clean.loc[:,c] = matt_test_clean.loc[:,c].astype(float)


matt_test_clean.to_pickle(folder+'2021_Competition_Test_col_184_275_Matt.pkl')
matt_test_clean.to_csv(folder+'2021_Competition_Test_col_184_275_Matt.csv')
matt_clean_cols = matt_clean.dtypes.reset_index


string_list = [l for l in dict(matt_clean.dtypes) if matt_clean.dtypes[l] == 'object'][1:]
num_list = [l for l in dict(matt_clean.dtypes) if matt_clean.dtypes[l] in ['int64','float64']]

string_list = string_list[1:]

vc = []
for s in string_list:
    for i in range(matt_clean[s].value_counts().shape[0]):
        vc.append([s,matt_clean[s].value_counts().index[i],matt_clean[s].value_counts()[i]])

vc_df = pd.DataFrame(vc,columns=['Column','Category','Count'])
vc_df.to_excel(folder+'Training Set Value Counts.xlsx')

#Used to see unique values in the dirty columns



#Code to clean Matt's columns
matt_clean = matt_clean.replace('*',np.nan)
matt_test_clean = matt_test_clean.replace('*',np.nan)

#Convert to float
matt_test_clean[matt_dirty_test_columns] = matt_test_clean[matt_dirty_test_columns].astype(float)
matt_clean[matt_dirty_train_columns] = matt_clean[matt_dirty_train_columns].astype(float)

#Export to pickle file
matt_clean.to_pickle(folder+'2021_Competition_Train_col_184_275_Matt.pkl')
matt_clean.to_csv(folder+'2021_Competition_Train_col_184_275_Matt.pkl')
matt_clean = pd.read_pickle(folder+'2021_Competition_Train_col_184_275_Matt.pkl')


matt_test_clean.to_pickle(folder+'2021_Competition_Test_col_184_275_Matt.pkl')
#matt_test_clean = pd.read_pickle(folder+'2021_Competition_Test_col_184_275_Matt.pkl')
matt_test_clean.to_csv(folder+'2021_Competition_Test_col_184_275_Matt.csv')

#Gives mean, median, mode, range
desc_train = matt_clean.describe()
desc_test = matt_test_clean.describe()
desc_train.to_excel(folder+'Training Set Summary Statistics.xlsx')
vc = {}
for s in string_list:
    vc[s] = matt_test_clean[s].value_counts()

#Gives Data Types for each column
dtypes = matt_clean.dtypes
dtypes.to_excel(folder+"Matt Columns Data Types.xlsx")

zero_list = []

for i in range(desc_train.shape[1]):
    if desc_train.iloc[1,i] == 0:
        zero_list.append(desc_train.columns[i])
        
matt_columns_num = matt_clean


matt_clean.drop('Unnamed: 0',axis=1,inplace=True)
