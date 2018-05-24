#原始的文件大小10GB+，去除了一些列，存为pickle文件后数据大小缩小了好几倍，pandas读取速度飞快
import numpy as np
import pandas as pd
import time,datetime
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns',200)
import warnings
warnings.filterwarnings('ignore')

train = pd.read_table('../data/round2_train.txt',header=0,delim_whitespace=True)
test = pd.read_table('../data/round2_ijcai_18_test_a_20180425.txt',header=0,delim_whitespace=True)
test_b = pd.read_table('../data/round2_ijcai_18_test_b_20180510.txt',header=0,delim_whitespace=True)

all_df = pd.concat([train,test,test_b])
all_df.reset_index(drop=True,inplace=True)
all_df.drop(['item_category_list','item_property_list','predict_category_property'],axis=1,inplace=True)

def code_level(s):
    if s==-1:
        return -1
    else:
        a = str(s)[2:]
        a = int(a)
        return a

lbl = LabelEncoder()
feats = ['item_id','item_brand_id','item_city_id','user_id','context_id','shop_id']
for feat in feats:
    all_df[feat] = lbl.fit_transform(all_df[feat])
#item_brand_id有缺失,继续保留-1为缺失值
all_df['item_brand_id'] = all_df['item_brand_id']-1
level_feats = ['user_age_level','user_occupation_id',
'user_star_level','context_page_id','shop_star_level']
for feat in level_feats:
    all_df[feat] = all_df[feat].apply(code_level)

all_df[['context_timestamp1']] = all_df[['context_timestamp']].apply(lambda s:pd.to_datetime(s,unit='s'))
delta = datetime.timedelta(hours=8)
all_df['context_timestamp1'] = all_df['context_timestamp1'].apply(lambda s:s+delta)
all_df['day'] = all_df['context_timestamp1'].apply(lambda s:s.day)
all_df['hour'] = all_df['context_timestamp1'].apply(lambda s:s.hour)
all_df.drop(['context_timestamp1'],axis=1,inplace=True)
all_df.to_pickle('../data/row_feats_ex_list_ab.pkl')
