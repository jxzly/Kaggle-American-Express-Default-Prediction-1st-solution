import pandas as pd
import numpy as np
import os,random
import datetime
from contextlib import contextmanager
from tqdm import tqdm

from sklearn.metrics import roc_auc_score,mean_squared_error,average_precision_score,log_loss
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler

import torch.cuda.amp as amp

from scheduler import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='./input/')
parser.add_argument("--save_dir", type=str, default='tmp')
parser.add_argument("--use_apm", action='store_true', default=False)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--do_train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--remark", type=str, default='')

args, unknown = parser.parse_known_args()

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(args.seed)

id_name = 'customer_ID'
label_name = 'target'

os.makedirs('./output/',exist_ok=True)

gpus = list(range(torch.cuda.device_count()))
print('available gpus:',gpus)

@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def Metric(labels,preds):
    return amex_metric_mod(labels,preds)

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None


def Lgb_train_and_predict(train, test, config, gkf=False, aug=None, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + f'{args.save_dir}/'
    else:
        output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    os.system(f'cp ./*.py {output_path}')
    os.system(f'cp ./*.sh {output_path}')
    config['lgb_params']['seed'] = config['seed']
    oof, sub = None,None
    if train is not None:
        log = open(output_path + '/train.log','w',buffering=1)
        log.write(str(config)+'\n')
        features = config['feature_name']
        params = config['lgb_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = config['seed']
        oof = train[[id_name]]
        oof[label_name] = 0

        all_valid_metric,feature_importance = [],[]
        if gkf:
            tmp = train[[id_name,label_name]].drop_duplicates(id_name).reset_index(drop=True)
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            split = skf.split(tmp,tmp[label_name])
            new_split = []
            for trn_index, val_index in split:
                trn_uids = tmp.loc[trn_index,id_name].values
                val_uids = tmp.loc[val_index,id_name].values
                new_split.append((train.loc[train[id_name].isin(trn_uids)].index,train.loc[train[id_name].isin(val_uids)].index))
            split = new_split

            # skf = GroupKFold(n_splits=folds)
            # split = skf.split(train,train[label_name],train[id_name])
        else:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            split = skf.split(train,train[label_name])
        for fold, (trn_index, val_index) in enumerate(split):
            evals_result_dic = {}
            train_cids = train.loc[trn_index,id_name].values
            if aug:
                train_aug = aug.loc[aug[id_name].isin(train_cids)]
                trn_data = lgb.Dataset(train.loc[trn_index,features].append(train_aug[features]), label=train.loc[trn_index,label_name].append(train_aug[label_name]))
            else:
                trn_data = lgb.Dataset(train.loc[trn_index,features], label=train.loc[trn_index,label_name])

            val_data = lgb.Dataset(train.loc[val_index,features], label=train.loc[val_index,label_name])
            model = lgb.train(params,
                train_set  = trn_data,
                num_boost_round   = rounds,
                valid_sets = [trn_data,val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose
            )
            model.save_model(output_path + '/fold%s.ckpt'%fold)

            valid_preds = model.predict(train.loc[val_index,features], num_iteration=model.best_iteration)
            oof.loc[val_index,label_name] = valid_preds

            for i in range(len(evals_result_dic['valid_1'][params['metric']])//verbose):
                Write_log(log,' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose,evals_result_dic['training'][params['metric']][i*verbose],evals_result_dic['valid_1'][params['metric']][i*verbose]))
            all_valid_metric.append(Metric(train.loc[val_index,label_name],valid_preds))
            Write_log(log,'- fold%s valid metric: %.6f\n'%(fold,all_valid_metric[-1]))

            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            feature_importance.append(pd.DataFrame({'feature_name':feature_name,'importance_gain':importance_gain,'importance_split':importance_split}))

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
        feature_importance_df.to_csv(output_path + '/feature_importance.csv',index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        global_valid_metric = Metric(train[label_name].values,oof[label_name].values)
        Write_log(log,'all valid mean metric:%.6f, global valid metric:%.6f'%(mean_valid_metric,global_valid_metric))

        oof.to_csv(output_path + '/oof.csv',index=False)

        log.close()
        os.rename(output_path + '/train.log', output_path + '/train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'mean metric':[round(mean_valid_metric,6)],'global metric':[round(global_valid_metric,6)],'remark':[args.remark]})
        if not os.path.exists(output_root + '/experiment_log.csv'):
            log_df.to_csv(output_root + '/experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + '/experiment_log.csv',index=False,header=None,mode='a')

    if test is not None:
        sub = test[[id_name]]
        sub['prediction'] = 0
        for fold in range(folds):
            model = lgb.Booster(model_file=output_path + '/fold%s.ckpt'%fold)
            test_preds = model.predict(test[features], num_iteration=model.best_iteration)
            sub['prediction'] += (test_preds / folds)
        sub[[id_name,'prediction']].to_csv(output_path + '/submission.csv.zip', compression='zip',index=False)
    if args.save_dir in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub,(mean_valid_metric,global_valid_metric)

class TaskDataset:
    def __init__(self,df_series,df_feature,uidxs,df_y=None):
        self.df_series = df_series
        self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1,i2,idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1,1:].values

        if len(series.shape) == 1:
            series = series.reshape((-1,)+series.shape[-1:])
        series_ = series.copy()
        series_[series_!=0] = 1.0 - series_[series_!=0] + 0.001
        feature = self.df_feature.loc[idx].values[1:]
        feature_ = feature.copy()
        feature_[feature_!=0] = 1.0 - feature_[feature_!=0] + 0.001
        if self.df_y is not None:
            label = self.df_y.loc[idx,[label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    }
        else:
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    'FEATURE': np.concatenate([feature,feature_]),
                    }

    def collate_fn(self, batch):
        """
        Padding to same size.
        """

        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros((batch_size, 1))

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            v = item['FEATURE'].astype(np.float32)
            batch_feature[i] = torch.tensor(v).float()
            if self.df_y is not None:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()

        return {'batch_series':batch_series,'batch_mask':batch_mask,'batch_feature':batch_feature,'batch_y':batch_y}


def NN_train_and_predict(train, test, model_class, config, use_series_oof, logit=False, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + f'{args.save_dir}/'
    else:
        output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    os.system(f'cp ./*.py {output_path}')
    feature_name = config['feature_name']
    obj_max = config['obj_max']
    epochs = config['epochs']
    smoothing = config['smoothing']
    patience = config['patience']
    lr = config['lr']
    batch_size = config['batch_size']
    folds = config['folds']
    seed = config['seed']
    if train is not None:
        train_series,train_feature,train_y,train_series_idx = train

        oof = train_y[[id_name]]
        oof['fold'] = -1
        oof[label_name] = 0.0
        oof[label_name] = oof[label_name].astype(np.float32)
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w',buffering=1)
        log.write(str(config)+'\n')

        all_valid_metric = []

        skf = StratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)

        model_num = 0
        train_folds = []

        for fold, (trn_index, val_index) in enumerate(skf.split(train_y,train_y[label_name])):

            train_dataset = TaskDataset(train_series,train_feature,[train_series_idx[i] for i in trn_index],train_y)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)
            valid_dataset = TaskDataset(train_series,train_feature,[train_series_idx[i] for i in val_index],train_y)
            valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False, drop_last=False, collate_fn=valid_dataset.collate_fn,num_workers=args.num_workers)

            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            scheduler = Adam12()

            model.cuda()
            if args.use_apm:
                scaler = amp.GradScaler()
            optimizer = scheduler.schedule(model, 0, epochs)[0]

            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
            #                                                 max_lr=1e-2, epochs=epochs, steps_per_epoch=len(train_dataloader))
            #torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=lr, weight_decay=0.00001,eps=1e-5)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])


            loss_tr = nn.BCELoss()
            loss_tr1 = nn.BCELoss(reduction='none')
            if obj_max == 1:
                best_valid_metric = 0
            else:
                best_valid_metric = 1e9
            not_improve_epochs = 0
            if args.do_train:
                for epoch in range(epochs):
                    # if epoch <= 13:
                    #     continue
                    np.random.seed(666*epoch)
                    train_loss = 0.0
                    train_num = 0
                    scheduler.step(model,epoch,epochs)
                    model.train()
                    bar = tqdm(train_dataloader)
                    for data in bar:
                        optimizer.zero_grad()
                        for k in data:
                            data[k] = data[k].cuda()
                        y = data['batch_y']
                        if args.use_apm:
                            with amp.autocast():
                                outputs = model(data)
                                # loss_series = loss_tr1(series_outputs,y.repeat(1,13))
                                # loss_series = (loss_series * data['batch_mask']).sum() / data['batch_mask'].sum()
                                # if epoch < 30:
                                #     loss = loss_series
                                # else:
                                loss = loss_tr(outputs,y) #+ loss_series # 0.5 * (loss_tr(outputs,y) + loss_feature(feature,y))
                            if str(loss.item()) == 'nan': continue
                            scaler.scale(loss).backward()
                            torch.nn.utils.clip_grad_norm(model.parameters(), clipnorm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(data)
                            loss = loss_tr(outputs,y)
                            loss.backward()
                            optimizer.step()
                        # scheduler.step()
                        train_num += data['batch_feature'].shape[0]
                        train_loss += data['batch_feature'].shape[0] * loss.item()
                        bar.set_description('loss: %.4f' % (loss.item()))

                    train_loss /= train_num

                    # eval
                    model.eval()
                    valid_preds = []
                    for data in tqdm(valid_dataloader):
                        for k in data:
                            data[k] = data[k].cuda()
                        with torch.no_grad():
                            if logit:
                                outputs = model(data).sigmoid()
                                # feature,outputs = model(data)
                                # outputs = outputs.sigmoid()
                            else:
                                outputs = model(data)
                                # feature,outputs = model(data)
                        valid_preds.append(outputs.detach().cpu().numpy())

                    valid_preds = np.concatenate(valid_preds).reshape(-1)
                    valid_Y = train_y.loc[val_index,label_name].values # oof train
                    valid_mean = np.mean(valid_preds)
                    valid_metric = Metric(valid_Y,valid_preds)

                    if obj_max*(valid_metric) > obj_max*best_valid_metric:
                        if len(gpus) > 1:
                            torch.save(model.module.state_dict(),output_path + 'fold%s.ckpt'%fold)
                        else:
                            torch.save(model.state_dict(),output_path + 'fold%s.ckpt'%fold)
                        not_improve_epochs = 0
                        best_valid_metric = valid_metric
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,valid_mean))
                    else:
                        not_improve_epochs += 1
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,valid_mean,not_improve_epochs))
                        if not_improve_epochs >= patience:
                            break

            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda' if torch.cuda.is_available() else 'cpu') )

            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            model.cuda()
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()

            valid_preds = []
            valid_Y = []
            for data in tqdm(valid_dataloader):
                for k in data:
                    data[k] = data[k].cuda()
                with torch.no_grad():
                    if logit:
                        outputs = model(data).sigmoid()
                        # feature,outputs = model(data)
                        # outputs = outputs.sigmoid()
                    else:
                        outputs = model(data)
                        # feature,outputs = model(data)
                valid_preds.append(outputs.detach().cpu().numpy())
                valid_Y.append(y.detach().cpu().numpy())

            valid_preds = np.concatenate(valid_preds).reshape(-1)
            valid_Y = train_y.loc[val_index,label_name].values # oof train
            valid_mean = np.mean(valid_preds)
            valid_metric = Metric(valid_Y,valid_preds)
            Write_log(log,'[fold %s] best_valid_metric: %.6f, best_valid_mean: %.6f'%(fold,valid_metric,valid_mean))

            all_valid_metric.append(valid_metric)
            oof.loc[val_index,label_name] = valid_preds
            oof.loc[val_index,'fold'] = fold
            train_folds.append(fold)

        mean_valid_metric = np.mean(all_valid_metric)
        Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))
        oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv',index=False)

        if test is None:
            log.close()
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'folds':folds,'metric':[round(mean_valid_metric,6)],'lb':[np.nan],'remark':[config['remark']]})
        if not os.path.exists(output_root + 'experiment_log.csv'):
            log_df.to_csv(output_root + 'experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + 'experiment_log.csv',index=False,mode='a',header=None)

    if test is not None:
        if train is None:
            log = open(output_path + 'test.log','w', buffering=1)
            Write_log(log,str(config)+'\n')
        test_series,test_feature,test_series_idx = test

        sub = test_feature[-len(test_series_idx):][[id_name]].reset_index(drop=True)
        sub['prediction'] = 0

        test_dataset = TaskDataset(test_series,test_feature,test_series_idx)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)
        models = []
        for fold in range(folds):
            if not os.path.exists(output_path + 'fold%s.ckpt'%fold):
                continue
            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            model.cuda()
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda') )
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()
            models.append(model)
        print('model count:',len(models))
        test_preds = []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                for k in data:
                    data[k] = data[k].cuda()

                if logit:
                    # outputs = model(data).sigmoid()
                    outputs = torch.stack([m(data).sigmoid() for m in models],0).mean(0)
                    # feature,outputs = model(data)
                    # outputs = outputs.sigmoid()
                else:
                    # outputs = model(data)
                    outputs = torch.stack([m(data) for m in models],0).mean(0)
                    # feature,outputs = model(data)
                test_preds.append(outputs.cpu().detach().numpy())
        test_preds = np.concatenate(test_preds).reshape(-1)
        test_mean = np.mean(test_preds)
        Write_log(log,'test_mean: %.6f'%(test_mean))
        sub['prediction'] = test_preds
        sub.to_csv(output_path+'submission.csv.zip',index=False, compression='zip')
    else:
        sub = None

    if args.save_dir in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub
