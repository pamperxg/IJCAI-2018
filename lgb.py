import numpy as np
import pandas as pd
import time,datetime
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns',200)
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from com_utils import *
import gc
all_df = pd.read_pickle('../data/row_feats_ex_list_ab.pkl')
test_b = pd.read_table('../data/round2_ijcai_18_test_b_20180510.txt',header=0,delim_whitespace=True)
all_df.loc[all_df['day']==31,'day'] = 0


--------------------------------------------------------------------------------------------------------------
#一些提取特征方法
#前面一次，后面一次搜索的间隔
def do_next_Query( df,agg_suffix='nextQuery', agg_type='float32'):
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_QUERY = [
    {'groupby': ['user_id']},
    {'groupby': ['user_id', 'item_id']},
    {'groupby': ['user_id', 'shop_id']},
    {'groupby': ['user_id', 'item_brand_id']},
    {'groupby': ['user_id', 'shop_id','item_brand_id']}
]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_QUERY:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['context_timestamp']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).context_timestamp.shift(-1) - df.context_timestamp).astype(agg_type)
        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Query( df,agg_suffix='prevQuery', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    GROUP_BY_PRE_QUERY = [
    {'groupby': ['user_id']},
    {'groupby': ['user_id', 'item_id']},
    {'groupby': ['user_id', 'shop_id']},
    {'groupby': ['user_id', 'item_brand_id']},
    {'groupby': ['user_id', 'shop_id','item_brand_id']}
]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_PRE_QUERY:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['context_timestamp']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.context_timestamp - df[all_features].groupby(spec[
                'groupby']).context_timestamp.shift(+1) ).astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)

#历史一段时间内的搜索、转化数
def get_pre_conver_query(df,train_day,day_gap):
    train_df = df[df.day==train_day]
    feats_df = df[(df.day<train_day)&(df.day>=train_day-day_gap)]
    CONVER_QUERY = [
            ['user_id'],
            ['item_id'],
            ['shop_id'],
            ['item_brand_id'],
            ['shop_id','item_brand_id'],
            ['user_id', 'item_id'],
            ['user_id', 'shop_id'],
            ['user_id', 'item_brand_id'],
            ['user_id', 'shop_id','item_brand_id']
    ]
    for cols in CONVER_QUERY:
        query_feats = '{}_{}_{}'.format('_'.join(cols),day_gap,'query')
        conver_feats = '{}_{}_{}'.format('_'.join(cols),day_gap,'conver')
        groupby_object = feats_df.groupby(cols)
        train_df = train_df.merge(groupby_object['is_trade'].size().reset_index().\
                                  rename(columns={'is_trade':query_feats}),on=cols,how='left')
        train_df = train_df.merge(groupby_object['is_trade'].sum().reset_index().\
                                  rename(columns={'is_trade':conver_feats}),on=cols,how='left')
        predictors.append(query_feats)
        predictors.append(conver_feats)
    return train_df

def do_count( df, group_cols, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

#cumcount,分组里面index顺序
def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

#一些加了没什么用的统计特征
# def get_his_stats_feat(df,train_day,day_gap):
#     train_df = df[df.day==train_day]
#     feats_df = df[(df.day<train_day)&(df.day>=train_day-day_gap)]
#     MEAN_FE = [
#         ['user_id','context_page_id'],
#         ['user_id','item_collected_level'],
#         ['user_id','item_price_level'],
#         ['user_id','item_pv_level'],
#         ['user_id','item_sales_level'],
#         ['user_id','shop_review_num_level'],
#         ['user_id','shop_review_positive_rate'],
#         ['user_id','shop_score_delivery'],
#         ['user_id','shop_score_service'],
#         ['user_id','shop_score_description'],
#         ['user_id','shop_star_level'],
#         ['item_id','context_page_id'],
#         ['item_id','user_age_level'],
#         ['item_id','user_star_level'],
#         ['shop_id','context_page_id'],
#         ['shop_id','user_age_level'],
#         ['shop_id','user_star_level'],
#         ['item_brand_id','user_age_level'],
#         ['item_brand_id','user_star_level']
#     ]
#     VAR_FE = [
#         ['user_id','context_page_id'],
#         ['user_id','item_collected_level'],
#         ['user_id','item_price_level'],
#         ['user_id','item_pv_level'],
#         ['user_id','item_sales_level'],
#         ['user_id','shop_review_num_level'],
#         ['user_id','shop_review_positive_rate'],
#         ['user_id','shop_score_delivery'],
#         ['user_id','shop_score_service'],
#         ['user_id','shop_score_description'],
#         ['user_id','shop_star_level'],
#         ['item_id','context_page_id'],
#         ['item_id','user_age_level'],
#         ['item_id','user_star_level'],
#         ['shop_id','context_page_id'],
#         ['shop_id','user_age_level'],
#         ['shop_id','user_star_level'],
#         ['item_brand_id','user_age_level'],
#         ['item_brand_id','user_star_level']

#     ]
#     UNIQ_FE = [
#         ['user_id','item_id'],
#         ['user_id','shop_id'],
#         ['user_id','item_brand_id']
#     ]
#     for cols in MEAN_FE:
#         feats = '{}_{}_{}'.format('_'.join(cols),day_gap,'mean')
#         print("\nGen",feats)
#         groupby_object = feats_df.groupby([cols[0]])
#         train_df = train_df.merge(groupby_object[cols[1]].mean().reset_index().\
#                                  rename(columns={cols[1]:feats}),on=cols[0],how='left')
#         predictors.append(feats)
#     for cols in VAR_FE:
#         feats = '{}_{}_{}'.format('_'.join(cols),day_gap,'var')
#         print("\nGen",feats)
#         groupby_object = feats_df.groupby([cols[0]])
#         train_df = train_df.merge(groupby_object[cols[1]].var().reset_index().\
#                                  rename(columns={cols[1]:feats}),on=cols[0],how='left')
#         predictors.append(feats)

#     for cols in UNIQ_FE:
#         feats = '{}_{}_{}'.format('_'.join(cols),day_gap,'uniq')
#         print("\nGen",feats)
#         groupby_object = feats_df.groupby([cols[0]])
#         train_df = train_df.merge(groupby_object[cols[1]].nunique().reset_index().\
#                                  rename(columns={cols[1]:feats}),on=cols[0],how='left')
#         predictors.append(feats)
#     return train_df

--------------------------------------------------------------------------------------------------------------
#特征生成
day7 = all_df[all_df.day==7]
day7 = day7.sort_values(by='context_timestamp')
#user下一次搜索时间
predictors=[]
day7 = do_next_Query(day7)
#user下一次搜索时间
predictors=[]
day7 = do_prev_Query(day7)
#前两天的搜索数转化数
predictors=[]
pre2day = get_pre_conver_query(all_df,7,2)
pre2day = pre2day.sort_values(by='context_timestamp')
his_2day_conver_num = pre2day[predictors]
#前7天的搜索数转化数
predictors=[]
pre7day = get_pre_conver_query(all_df,7,7)
pre7day = pre7day.sort_values(by='context_timestamp')
his_7day_conver_num = pre7day[predictors]
#得到整个的搜索数转化数特征
his_query_conver_num = pd.concat([his_2day_conver_num,his_7day_conver_num],axis=1)

CUMCOUNT_QUERY = [
            ['user_id'],
            ['user_id', 'item_id'],
            ['user_id', 'shop_id'],
            ['user_id', 'item_brand_id'],
            ['user_id','shop_id','item_brand_id']
    ]
for col in CUMCOUNT_QUERY:
    #一天第几次出现
    day7 = do_cumcount(day7,col,'context_timestamp')
    #一天出现多少次
    day7 = do_count(day7,col)
day7 = pd.concat([day7,his_query_conver_num],axis=1)

#一些统计特征
# predictors=[]
# stats_feats = get_his_stats_feat(all_df,7,7)
# stats_feats = stats_feats.sort_values(by='context_timestamp')
# stats_feats = stats_feats[predictors]
# day7 = pd.concat([day7,stats_feats],axis=1)

--------------------------------------------------------------------------------------------------------------
#模型训练测试
train = day7[day7.is_trade.notnull()]
test = day7[day7.is_trade.isnull()]
ex_list = ['context_timestamp','instance_id','is_trade','day','hour']
features = [i for i in list(train.columns) if i not in ex_list]
print('特征数目：',len(features))  #77

train_df = train[train.hour<10]
# train_df = pd.concat([train_df,day4])
val_df = train[train.hour>=10]

X_train = train_df[features]
y_train = train_df['is_trade']
X_test = val_df[features]
y_test = val_df['is_trade']

params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'binary_logloss',
        'learning_rate': 0.04,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0.99,  # L1 regularization term on weights
        'reg_lambda': 0.9,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 1,
    }

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=lgb_eval)

train_x = train[features]
train_y = train['is_trade']
test_a = test[features]

lgb_train_a = lgb.Dataset(train_x, train_y)
gbm_a = lgb.train(params,
                lgb_train_a,
                valid_sets=lgb_train_a,
                num_boost_round=3000,
                early_stopping_rounds=50
)

--------------------------------------------------------------------------------------------------------------
#生成提交结果
y_pred = gbm_a.predict(test_a, num_iteration=gbm_a.best_iteration)
a_pred = test[['instance_id']]
a_pred['predicted_score'] = y_pred
sub = test_b[['instance_id']]
sub = pd.merge(sub,a_pred,on='instance_id',how='left')
sub.to_csv('../sub/sub.txt',index=None,sep=' ')
