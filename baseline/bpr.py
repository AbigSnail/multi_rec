import cornac
from cornac.data import Dataset
from collections import OrderedDict
import pandas as pd
from  numpy import *
import numpy as np
pathtrans="transactions_train.csv"
image_features_file = "res.csv"
def hit_ratio(y_true, y_pred):
    hits = 0
    for true_item  in   y_pred :
        if true_item in y_true:
            hits += y_true.count(true_item)
    hit_ratio = hits/len(y_true)
    pre_ratio=hits/len(y_pred)
    return hit_ratio,pre_ratio

def huafen(data):

    interactions_df=data
    user_ids = interactions_df['customer_id'].unique()
    item_ids = interactions_df['article_id'].unique()
    num_users = len(set(interactions_df['customer_id']))
    num_items = len(set(interactions_df['article_id']))
    # Creating a dictionary mapping user IDs to integers starting from 0
    uid_map = OrderedDict((user_id, user_id) for index, user_id in enumerate(interactions_df['customer_id']))
    iid_map = OrderedDict((item_id, item_id) for index, item_id in enumerate(interactions_df['article_id']))
    user_indices = np.array(interactions_df['customer_id'])
    item_indices = np.array(interactions_df['article_id'])
    rating_values = np.array([1]*len(interactions_df['customer_id']))
    dataset = Dataset(num_users=num_users, num_items=num_items, uid_map=uid_map, iid_map=iid_map,
                      uir_tuple=(user_indices, item_indices, rating_values))
    return dataset


def readdata(path):
    df = pd.read_csv(path)
    grouped_data = df.groupby('customer_id')
    train_data = grouped_data.apply(lambda x: x.head(40)).reset_index(drop=True)
    test_data = df.drop(train_data.index)
    return train_data,test_data


#构建训练集和测试集
train,test=readdata(pathtrans)
test=test[test['article_id'].isin(train['article_id'])]

#获取图像特征
image_features_columns = ['id']
collist=[str(i+1) for i in range(64) ]
image_features_columns=image_features_columns+collist
image_features_df = pd.read_csv(image_features_file)
#重新选择feature
image_features_df = image_features_df[image_features_df['id'].isin(train['article_id'])]
#对所有的id重新进行映射
item_map = OrderedDict((item, index) for index, item in enumerate(image_features_df['id']))
user_map = OrderedDict((user, index) for index, user in enumerate(train['customer_id'].unique()))

# 使用映射将列表中的值替换为索引
image_features_df['id'] = [item_map[item] for item in image_features_df['id']]
train['article_id'] = [item_map[item] for item in  train['article_id']]
test['article_id'] = [item_map[item] for item in  test['article_id']]
train['customer_id'] = [user_map[item] for item in  train['customer_id']]
test['customer_id'] = [user_map[item] for item in  test['customer_id']]

traindata=huafen(train)
testdata=test

bpr=cornac.models.BPR( k=20, verbose=True,max_iter=100)
bpr.fit(traindata)


result=[]
hr_list=[]
pr_list=[]
grouped = testdata.groupby('customer_id')
# 遍历每个分组并进行进一步操作
i=0
for uid, group in grouped:
    # 在此处可以对每个分组进行操作
    # 例如，打印分组的内容

    my_list = bpr.score(uid, )
    sorted_indices = sorted(range(len(my_list)), key=lambda x: my_list[x])[:20]
    hr_list.append(hit_ratio(group['article_id'].tolist(),sorted_indices)[0])
    pr_list.append(hit_ratio(group['article_id'].tolist(), sorted_indices)[1])

hr_list.sort(reverse=True) 
pr_list.sort(reverse=True)
print(hr_list)
print(mean(hr_list))
print(pr_list)
print(mean(pr_list))