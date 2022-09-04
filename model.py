import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math

class Amodel(nn.Module):
    def __init__(self, series_dim, feature_dim, target_num, hidden_num, hidden_dim, drop_rate=0.5, use_series_oof=False):
        super(Amodel, self).__init__()
        self.use_series_oof = use_series_oof
        self.input_series_block = nn.Sequential(
                                        nn.Linear(series_dim, hidden_dim)
                                        ,nn.LayerNorm(hidden_dim)
                                        )
        self.input_feature_block = nn.Sequential(
                                        nn.Linear(feature_dim, hidden_dim)
                                        ,nn.BatchNorm1d(hidden_dim)
                                        ,nn.LeakyReLU()
                                        )
        self.gru_series = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_feature_block = []
        for h in range(hidden_num-1):
            self.hidden_feature_block.extend([
                                     nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.BatchNorm1d(hidden_dim)
                                     ,nn.Dropout(drop_rate)
                                     ,nn.LeakyReLU()
                                     ])
        self.hidden_feature_block = nn.Sequential(*self.hidden_feature_block)

        self.output_block = nn.Sequential(
                                         nn.Linear(3*hidden_dim if use_series_oof else 2*hidden_dim, 1*hidden_dim)
                                         ,nn.LeakyReLU()

                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.LeakyReLU()
                                         
                                         ,nn.Linear(1*hidden_dim, target_num)
                                         ,nn.Sigmoid()
                                         )

    def batch_gru(self,series,mask):
        node_num = mask.sum(dim=-1).detach().cpu()
        pack = nn.utils.rnn.pack_padded_sequence(series, node_num, batch_first=True, enforce_sorted=False)
        message,hidden = self.gru_series(pack)
        pooling_feature = []

        for i,n in enumerate(node_num.numpy()):
            n = int(n)
            bi = 0

            si = message.unsorted_indices[i]
            for k in range(n):

                if k == n-1:
                    sample_feature = message.data[bi+si]
                bi = bi + message.batch_sizes[k]

            pooling_feature.append(sample_feature)
        return torch.stack(pooling_feature,0)

    def forward(self, data):
        x1 = self.input_series_block(data['batch_series'])

        x1 = self.batch_gru(x1,data['batch_mask'])
        if self.use_series_oof:
            x2 = self.input_feature_block(data['batch_feature'])
            x2 = self.hidden_feature_block(x2)
            x = torch.cat([x1,x2],axis=1)
            y = self.output_block(x)
        else:
            y = self.output_block(x1)
        return y
