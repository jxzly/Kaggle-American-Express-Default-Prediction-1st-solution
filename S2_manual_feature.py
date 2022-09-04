import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool


def one_hot_encoding(df,cols,is_drop=True):
    for col in cols:
        print('one hot encoding:',col)
        dummies = pd.get_dummies(pd.Series(df[col]),prefix='oneHot_%s'%col)
        df = pd.concat([df,dummies],axis=1)
    if is_drop:
        df.drop(cols,axis=1,inplace=True)
    return df

def cat_feature(df):
    one_hot_features = [col for col in df.columns if 'oneHot' in col]
    if lastk is None:
        num_agg_df = df.groupby("customer_ID",sort=False)[one_hot_features].agg(['mean', 'std', 'sum', 'last'])
    else:
        num_agg_df = df.groupby("customer_ID",sort=False)[one_hot_features].agg(['mean', 'std', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

    if lastk is None:
        cat_agg_df = df.groupby("customer_ID",sort=False)[cat_features].agg(['last', 'nunique'])
    else:
        cat_agg_df = df.groupby("customer_ID",sort=False)[cat_features].agg(['nunique'])
    cat_agg_df.columns = ['_'.join(x) for x in cat_agg_df.columns]

    count_agg_df = df.groupby("customer_ID",sort=False)[['S_2']].agg(['count'])
    count_agg_df.columns = ['_'.join(x) for x in count_agg_df.columns]
    df = pd.concat([num_agg_df, cat_agg_df,count_agg_df], axis=1).reset_index()
    print('cat feature shape after engineering', df.shape )

    return df

def num_feature(df):
    if num_features[0][:5] == 'rank_':
        num_agg_df = df.groupby("customer_ID",sort=False)[num_features].agg(['last'])
    else:
        if lastk is None:
            num_agg_df = df.groupby("customer_ID",sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
        else:
            num_agg_df = df.groupby("customer_ID",sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
    if num_features[0][:5] != 'rank_':
        for col in num_agg_df.columns:
            num_agg_df[col] = num_agg_df[col] // 0.01
    df = num_agg_df.reset_index()
    print('num feature shape after engineering', df.shape )

    return df

def diff_feature(df):
    diff_num_features = [f'diff_{col}' for col in num_features]
    cids = df['customer_ID'].values
    df = df.groupby('customer_ID')[num_features].diff().add_prefix('diff_')
    df.insert(0,'customer_ID',cids)
    if lastk is None:
        num_agg_df = df.groupby("customer_ID",sort=False)[diff_num_features].agg(['mean', 'std', 'min', 'max', 'sum', 'last'])
    else:
        num_agg_df = df.groupby("customer_ID",sort=False)[diff_num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
    for col in num_agg_df.columns:
        num_agg_df[col] = num_agg_df[col] // 0.01

    df = num_agg_df.reset_index()
    print('diff feature shape after engineering', df.shape )

    return df

n_cpu = 16
transform = [['','rank_','ym_rank_'],[''],['']]

for li, lastk in enumerate([None,3,6]):
    for prefix in transform[li]:
        df = pd.read_feather(f'./input/train.feather').append(pd.read_feather(f'./input/test.feather')).reset_index(drop=True)
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        for col in [col for col in df.columns if 'S_' in col or 'P_' in col]:
            if col != 'S_2':
                df[col] = df[col].fillna(0)

        if lastk is not None:
            prefix = f'last{lastk}_' + prefix
            print('all df shape',df.shape)
            df['rank'] = df.groupby('customer_ID')['S_2'].rank(ascending=False)
            df = df.loc[df['rank']<=lastk].reset_index(drop=True)
            df = df.drop(['rank'],axis=1)
            print(f'last {lastk} shape',df.shape)

        if prefix == 'rank_':
            cids = df['customer_ID'].values
            df = df.groupby('customer_ID')[num_features].rank(pct=True).add_prefix('rank_')
            df.insert(0,'customer_ID',cids)
            num_features = [f'rank_{col}' for col in num_features]

        if prefix == 'ym_rank_':
            cids = df['customer_ID'].values
            df['ym'] = df['S_2'].apply(lambda x:x[:7])
            df = df.groupby('ym')[num_features].rank(pct=True).add_prefix('ym_rank_')
            num_features = [f'ym_rank_{col}' for col in num_features]
            df.insert(0,'customer_ID',cids)

        if prefix in ['','last3_']:
            df = one_hot_encoding(df,cat_features,False)

        vc = df['customer_ID'].value_counts(sort=False).cumsum()
        batch_size = int(np.ceil(len(vc) / n_cpu))
        dfs = []
        start = 0
        for i in range(min(n_cpu,int(np.ceil(len(vc) / batch_size)))):
            vc_ = vc[i*batch_size:(i+1)*batch_size]
            dfs.append(df[start:vc_[-1]])
            start = vc_[-1]

        pool = ThreadPool(n_cpu)

        if prefix in ['','last3_']:
            cat_feature_df = pd.concat(pool.map(cat_feature,tqdm(dfs,desc='cat_feature'))).reset_index(drop=True)

            cat_feature_df.to_feather(f'./input/{prefix}cat_feature.feather')

        if prefix in ['','last3_','last6_','rank_','ym_rank_']:
            num_feature_df = pd.concat(pool.map(num_feature,tqdm(dfs,desc='num_feature'))).reset_index(drop=True)
            num_feature_df.to_feather(f'./input/{prefix}num_feature.feather')

        if prefix in ['','last3_']:
            diff_feature_df = pd.concat(pool.map(diff_feature,tqdm(dfs,desc='diff_feature'))).reset_index(drop=True)
            diff_feature_df.to_feather(f'./input/{prefix}diff_feature.feather')

        pool.close()
