import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
import numpy as np
import pandas as pd
import cornac
from cornac.data.image import ImageModality
from cornac.data import Dataset
from cornac.eval_methods import RatioSplit
from cornac.metrics import Precision
from cornac.data import Reader
from collections import OrderedDict


VBPR=cornac.models.vbpr.recom_vbpr.VBPR(name='VBPR', k=10, k2=10, n_epochs=50, batch_size=100, learning_rate=0.005,
             lambda_w=0.01, lambda_b=0.01, lambda_e=0.0, use_gpu=False, trainable=True, verbose=True, init_params=None, seed=None)

pathtrans='../data/transactions_train3.csv'
def hit_ratio(y_true, y_pred):



    hits = 0
    for true_item  in   y_pred :
        if true_item in y_true:
            hits += y_true.count(true_item)
    hit_ratio = hits/len(y_true)
    pre_ratio=hits/len(y_pred)
    return hit_ratio,pre_ratio

def huafen(data):


    # interactions_columns=data
    # interactions_columns = ['customer_id', 'article_id']

    # 加载交互记录数据为Pandas DataFrame
    # interactions_df = pd.read_csv(interactions_file_path, usecols=interactions_columns)
    interactions_df=data
    # 3. 创建Cornac数据集

    # 创建用户和物品的ID列表
    user_ids = interactions_df['customer_id'].unique()
    item_ids = interactions_df['article_id'].unique()



    # 假设有3个用户和4个物品
    num_users = len(set(interactions_df['customer_id']))
    num_items = len(set(interactions_df['article_id']))


    # Creating a dictionary mapping user IDs to integers starting from 0
    uid_map = OrderedDict((user_id, index) for index, user_id in enumerate(interactions_df['customer_id']))
    iid_map = OrderedDict((item_id, item_id) for index, item_id in enumerate(interactions_df['article_id']))

    # 用户-物品-评分数据

    user_indices = np.array(interactions_df['customer_id'])
    item_indices = np.array(interactions_df['article_id'])
    rating_values = np.array([1]*len(interactions_df['customer_id']))

    # 创建Dataset对象


    dataset = Dataset(num_users=num_users, num_items=num_items, uid_map=uid_map, iid_map=iid_map,
                      uir_tuple=(user_indices, item_indices, rating_values))

    return dataset


import pandas as pd
def readdata(path):
    df = pd.read_csv(path)

    # 按照用户（uid）进行分组，并对每个用户的交互记录按时间排序
    # df_sorted = df.sort_values(by=['customer_id'])

    # 按照用户（uid）进行分组，并获取每个用户的前20个交互记录
    train_data = df.groupby('customer_id').head(20)
    # train_data=df
    # 获取剩余的交互记录作为测试集
    test_data = df.drop(train_data.index)

    return train_data,test_data


train,test=readdata(pathtrans)

test=test[test['article_id'].isin(train['article_id'])]
# df_large_group = df[df['uid'].isin(large_group.index)]

image_features_file='res.csv'
image_features_columns = ['id']
collist=[str(i+1) for i in range(64) ]
image_features_columns=image_features_columns+collist

image_features_df = pd.read_csv(image_features_file)
#重新选择feature
image_features_df = image_features_df[image_features_df['id'].isin(train['article_id'])]
#对tran_id重新编码

# 使用有序字典将唯一值映射为从0开始的索引
item_map = OrderedDict((item, index) for index, item in enumerate(image_features_df['id']))

# 使用映射将列表中的值替换为索引
image_features_df['id'] = [item_map[item] for item in image_features_df['id']]
train['article_id'] = [item_map[item] for item in  train['article_id']]
test['article_id'] = [item_map[item] for item in  test['article_id']]

#对image_features_df——id重新编码
item_ids = image_features_df['id'].tolist()

image_features = image_features_df.drop('id', axis=1).values

image_modality = ImageModality(features=image_features, ids=item_ids)


traindata=huafen(train)
testdata=huafen(test)

traindata.item_image=image_modality

vbpr = cornac.models.VBPR(k=100, k2=100, n_epochs=50, batch_size=100, learning_rate=0.005, lambda_w=0.01, lambda_b=0.01, lambda_e=0.0, use_gpu=True, verbose=True)
vbpr.fit(traindata)

#算命中率
#算准确率
#排序
result=[]
hr_list=[]
pr_list=[]
grouped = test.groupby('customer_id')
# 遍历每个分组并进行进一步操作
i=0
for uid, group in grouped:
    # 在此处可以对每个分组进行操作
    # 例如，打印分组的内容

    my_list = vbpr.score(uid, )
    sorted_indices = sorted(range(len(my_list)), key=lambda x: my_list[x])[:20]
    hr_list.append(hit_ratio(group['article_id'].tolist(),sorted_indices)[0])
    pr_list.append(hit_ratio(group['article_id'].tolist(), sorted_indices)[1])

from  numpy import *
hr_list.sort(reverse=True)  # 降序
pr_list.sort(reverse=True)
print(hr_list[:200])
print(mean(hr_list[:200]))
print(pr_list[:200])
print(mean(pr_list[:200]))