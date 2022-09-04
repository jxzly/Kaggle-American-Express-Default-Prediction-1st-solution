import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import *
root = args.root
seed = args.seed


train = pd.read_feather(f'./input/train.feather')
test = pd.read_feather(f'./input/test.feather')

def one_hot_encoding(df,cols,is_drop=True):
    for col in cols:
        print('one hot encoding:',col)
        dummies = pd.get_dummies(pd.Series(df[col]),prefix='oneHot_%s'%col)
        df = pd.concat([df,dummies],axis=1)
    if is_drop:
        df.drop(cols,axis=1,inplace=True)
    return df
cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
eps = 1e-3


train_y =  pd.read_csv(f'{root}/train_labels.csv')
train = train.merge(train_y,how='left',on=id_name)

print(train.shape,test.shape)

lgb_config = {
    'lgb_params':{
                  'objective' : 'binary',
                  'metric' : 'binary_logloss',
                  'boosting': 'dart',
                  'max_depth' : -1,
                  'num_leaves' : 64,
                  'learning_rate' : 0.035,
                  'bagging_freq': 5,
                  'bagging_fraction' : 0.7,
                  'feature_fraction' : 0.7,
                  'min_data_in_leaf': 256,
                  'max_bin': 63,
                  'min_data_in_bin': 256,
                  # 'min_sum_heassian_in_leaf': 10,
                  'tree_learner': 'serial',
                  'boost_from_average': 'false',
                  'lambda_l1' : 0.1,
                  'lambda_l2' : 30,
                  'num_threads': 24,
                  'verbosity' : 1,
    },
    'feature_name':[col for col in train.columns if col not in [id_name,label_name,'S_2']],
    'rounds':4500,
    'early_stopping_rounds':100,
    'verbose_eval':50,
    'folds':5,
    'seed':seed
}


Lgb_train_and_predict(train,test,lgb_config,gkf=True,aug=None,run_id='LGB_with_series_feature')
