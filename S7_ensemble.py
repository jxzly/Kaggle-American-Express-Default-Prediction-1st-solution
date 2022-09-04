import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

p0 = pd.read_csv('./output/LGB_with_manual_feature/submission.csv.zip')
p1 = pd.read_csv('./output/LGB_with_manual_feature_and_series_oof/submission.csv.zip')
p2 = pd.read_csv('./output/NN_with_series_feature/submission.csv.zip')
p3 = pd.read_csv('./output/NN_with_series_and_all_feature/submission.csv.zip')

p0['prediction'] = p0['prediction']*0.3 + p1['prediction']*0.35 + p2['prediction']*0.15 + p3['prediction']*0.1

p0.to_csv('./output/final_submission.csv.zip',index=False, compression='zip')
