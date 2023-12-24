from transformers import BertTokenizer, BertConfig
from collections import OrderedDict, namedtuple
from typing import List, Union
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import os

# 显示所有列
pd.set_option('display.max_columns',None)
# 显示所有行
pd.set_option('display.max_rows',None)

path2='./data/res3.csv'

class DataWrapper(object):
    def __init__(self):
        self.user_indices = []
        self.trans_indices = []
        self.ground_truth = []
        self.index = dict()

    def append(self, user_indice: int,
               trans_indices: List[int],
               ground_truth_indices: List[int] = None):

        assert (user_indice not in self.index, f'UserIndice {user_indice} is already in, please use `set_value` function!')

        self.user_indices.append(user_indice)#user的索引，csv表中的数据
        self.trans_indices.append(trans_indices)
        if ground_truth_indices is not None:
            self.ground_truth.append(ground_truth_indices)

        self.index[user_indice] = len(self) - 1#user的索引下标0-n

    def set_value(self,
                  user_indice: int,
                  trans_indices: List[int],
                  ground_truth: List[int] = None):

        index = self.index[user_indice]
        self.user_indices[index] = user_indice
        self.trans_indices[index] = trans_indices
        if ground_truth is not None:
            self.ground_truth[index] = ground_truth

    #shuffle函数的作用是打乱数据，生成0到数据长度的索引，然后按照索引打乱用户索引和转换索引。
    #如果ground_truth不为空，也按照相同的索引打乱。
    def shuffle(self):
        # 生成0到数据长度的索引
        indices = list(range(len(self)))
        random.shuffle(indices)
        self.user_indices = [self.user_indices[i] for i in indices]
        self.trans_indices = [self.trans_indices[i] for i in indices]
        if self.ground_truth:
            self.ground_truth = [self.ground_truth[i] for i in indices]

    def __len__(self):
        return len(self.user_indices)


class RecData(object):
    #元组
    _sys_fields = (
        'id', 'desc', 'info', 'image', 'tfrecord',
        'profile', 'context', 'user', 'item', 'trans'
    )
    def __init__(self,
                 items: pd.DataFrame,
                 users: pd.DataFrame,
                 trans: pd.DataFrame,
                 #一个用于神经网络模型的配置字典
                 config: Union[dict, BertConfig],
                 #要从中加载特征的文件路径
                 feature_path: str = None,
                 #图像缩放的布尔标志（可选）
                 resize_image: bool = False):

        assert 'article_id' in items and 'customer_id' in users
        assert 'article_id' in trans and 'customer_id' in trans

        #创建了一个有序字典，用于映射items和users DataFrame的索引到它们各自的id
        self.item_feature_dict = OrderedDict()
        self.user_feature_dict = OrderedDict()
        self.trans_feature_dict = OrderedDict()
        self.train_wrapper = DataWrapper()
        self.test_wrapper = DataWrapper()
        self.config = config
        self.resize_image = resize_image
        self.items = items
        self.users = users
        self.trans = trans
        # load/learn features maps
        self.items.reset_index(drop=True, inplace=True)#删除原来的id,增减0-n新的id
        self.users.reset_index(drop=True, inplace=True)
        self.trans.reset_index(drop=True, inplace=True)

        # items and users use reseted index
        # self.item_index_map = OrderedDict([(id, i) for i, id in enumerate(self.items['article_id'])])
        # self.user_index_map = OrderedDict([(id, i) for i, id in enumerate(self.users['customer_id'])])
        # self.trans['article_id'] = self.trans['article_id'].map(self.item_index_map)
        # self.trans['customer_id'] = self.trans['customer_id'].map(self.user_index_map)

        if feature_path is not None:
            self.load_feature_dict(feature_path)
        else:
            self._learn_feature_dict()
        self.item_data = None
        self.user_data = None
        self.trans_data = None

    def prepare_features(self, tokenizer: BertTokenizer):#:是类型建议符
        # 检查是否已经处理过特征
        if not self._processed:
            print('Process item features ...', end='')
            # length + 1 for padding
            # 为每个 item 处理信息特征，将特征映射到 info 矩阵中
            info = np.zeros((len(self.items), len(self.info_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.item_feature_dict.items()):
                info[:, i] = self.items[key].map(feat_map)#把csv中的特征进行数字化，使用特征的index来替换cvs表格中的数据
            info = np.array(info)[:, [1,2,4,5,6,7,8,9]]
            x=self.items.pop('detail_desc').to_list()#有的商品描述为空，需要去除
            x = [str(element) for element in x]
            # 使用 tokenizer 处理 item 描述文本
            desc = tokenizer(
                x,
                max_length=self.config.get('max_desc_length', 8),
                truncation=True,
                padding='max_length',
                return_attention_mask=False,
                return_token_type_ids=False,
                return_tensors='np'
            )['input_ids']

            image_feature = pd.read_csv(path2)
            image_feature = (np.array(image_feature)).reshape(len(self.items), 8, 8)
            # image_feature=np.ones((len(self.items),8,8))
            self.item_data = {'info': info,'desc': desc, 'image': image_feature}
            print('Done!')
            print('Process user features ...', end='')
            # 为每个 user 处理信息特征，将特征映射到 profile 矩阵中
            profile = np.zeros((len(self.users), len(self.profile_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.user_feature_dict.items()):
                profile[:, i] = self.users[key].map(feat_map)
            self.user_data = {'profile': profile}
            print('Done!')
            print('Process transaction features ...', end='')
            # 为每个 transaction 处理信息特征，将特征映射到 context 矩阵中
            context = np.zeros((len(self.trans), len(self.context_size)), dtype=np.int32)
            for i, (key, feat_map) in enumerate(self.trans_feature_dict.items()):
                #i是0-4；key是article_id...;feat_map是字典，例如article_id后面的字典{'0.0': 0, '1.0': 1, '3.0': 2... }
                context[:, i] = self.trans[key].map(feat_map)#对trans里面的数据进行重新映射
            self.trans_data = {'context': context}
            # 添加填充的索引值
            # self.trans.loc[-1] = {'article_id': -1, 'customer_id': -1}  # for padding indice
            print('Done!')
        else:
            print("Features are aleady prepared.")

    @property#@property装饰器用来创建只读属性
    def _processed(self):
        ## 定义一个名为flag的布尔类型变量，初值为self.item_data是否为空的布尔值。
        flag = self.item_data is not None
        flag &= self.user_data is not None
        flag &= self.trans_data is not None
        return flag


    def prepare_train(self, test_users: list = None):
        if test_users is not None:
            test_users = [self.user_index_map[user] for user in test_users]
            #测试集合
            test_users = set(test_users)
        # 创建一个tqdm进度条，并设置进度条的总进度为self.trans中user列中不同用户的数量。
        with tqdm(total=len(self.trans['customer_id'].unique()), desc='Process training data') as pbar:
            ## 遍历self.trans中按用户分组的数据集。
            for user_idx, df in self.trans.groupby('customer_id'):#user_idx是主键，df是视图
                # 更新进度条的进度。
                pbar.update()
                # 获取用户的交互索引和交互物品序列的索引列表。
                trans_indices = df.index.to_list()
                item_indices = df['article_id'].to_list()
                # if len(trans_indices) < self.config.get('max_history_length', 32) or (test_users is not None and user_idx in test_users):#两种成为测试样本的情况：本身购买记录少；指定了测试用户
                #     #购买交互记录小于最大购买历史长度
                #     if len(df) < self.config.get('top_k', 10):#购买商品的个数小于10个
                #         self.test_wrapper.append(user_idx, trans_indices[:1], item_indices[1:])
                #     else:#购买的商品的个数大于10个，gound truth的大小设置为10
                #         self.test_wrapper.append(
                #             user_idx,
                #             trans_indices[:-self.config.get('top_k', 10)],#键值不存在时返回默认值，这里默认值为10
                #             item_indices[-self.config.get('top_k', 10):]
                #         )
                # else:
                #     # train sample
                #     cut_offset = max(len(trans_indices)-self.config.get('top_k', 10), self.config.get('max_history_length', 32))
                #     self.train_wrapper.append(user_idx, trans_indices[:cut_offset])#train数据集为什么没有ground truth？当然没有，第二个参数其实就是ground truth了
                #     if cut_offset < len(trans_indices):
                #         # cut off for test
                #         self.test_wrapper.append(user_idx, trans_indices[:cut_offset], item_indices[cut_offset:])

                    # train sample

                newlen=self.config.get('max_history_length', 32)+20
                if  len(trans_indices)>newlen:
                    # cut off for test
                    self.test_wrapper.append(user_idx, trans_indices[:self.config.get('max_history_length', 32)], item_indices[self.config.get('max_history_length', 32):])
                else:
                    self.train_wrapper.append(user_idx, trans_indices[:self.config.get('max_history_length',
                                                                                       32)])  # train数据集为什么没有ground truth？当然没有，第二个参数其实就是ground truth了

        # shuffle train samples
        self.train_wrapper.shuffle()
        print('Train samples: {}'.format(len(self.train_wrapper)))
        print('Test samples: {}'.format(len(self.test_wrapper)))

    @property
    def infer_wrapper(self):
        wrapper = DataWrapper()
        trans_data = dict(list(self.trans.groupby('customer_id')))
        for i in self.users.index:
            trans_indices = trans_data.get(i)
            if  trans_indices is  None:
                continue
            trans_indices = trans_indices.index.to_list()
            if(len(trans_indices)>10):
                wrapper.append(i, trans_indices)#用户user i，i的购买记录为trans_indices
        return wrapper

    @property
    def info_size(self):
        size = []
        for _, feat_map in self.item_feature_dict.items():
            size.append(len(feat_map))
        return size

    @property
    def profile_size(self):
        size = []
        for _, feat_map in self.user_feature_dict.items():
            size.append(len(feat_map))
        return size

    @property
    def context_size(self):
        size = []
        for _, feat_map in self.trans_feature_dict.items():
            size.append(len(feat_map))
        return size


    def _learn_feature_dict(self):
        info_features = sorted(set(self.items.columns) - set(['id']))
        if self.config.get('use_item_id'):
            info_features.append('id')
        for col in info_features:
            vals = set(self.items[col])
            self.item_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(vals)])#item_freature_dict是在一个字典，关键词时列名，每个关键词的值是一个有序字典，关键字是属性名，值是属性名的index

        profile_features = sorted(set(self.users.columns) - set(self._sys_fields))#columns获取列名
        if self.config.get('use_user_id'):
            profile_features.append('id')
        for col in profile_features:
            vals = set(self.users[col])
            self.user_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(vals)])

        context_features = sorted(set(self.trans.columns) - set(self._sys_fields))
        for col in context_features:
            vals = set(self.trans[col])
            self.trans_feature_dict[col] = OrderedDict(
                [(val, i) for i, val in enumerate(vals)])
        self._display_feature_info()#可以删除


    def _display_feature_info(self):
        info = []
        for feat, feat_map in self.item_feature_dict.items():
            info.append({'subject': 'item', 'feature': feat, 'size': len(feat_map)})
        for feat, feat_map in self.user_feature_dict.items():
            info.append({'subject': 'user', 'feature': feat, 'size': len(feat_map)})
        for feat, feat_map in self.trans_feature_dict.items():
            info.append({'subject': 'trans', 'feature': feat, 'size': len(feat_map)})
        info = pd.DataFrame(info, index=None)
        print(info)

    def train_dataset(self, batch_size: int = 8):
        assert self._processed
        '''
        将序列填充到相同的长度
        ·sequences:序列列表（每个序列都是整数列表）。
        ·maxlen :可以对此参数进行设定，代表所有序列的最大长度。如果没有提供默认为补齐到最长序列。
        ·dtype : 参数设置可供选择。可以让序列中的数以不同的进制显示
        ·padding:参数选择为“pre”和“post”，pre为在序列前进行拉伸或者截断，post是在序列最后进行拉伸或者截断
        ·truncating:参数可以选择为‘pre’或者‘post’在序列的开头或结尾从大于 maxlen 的序列中删除值。
        ·value :浮点或字符串，填充值。（可选，默认为0）
        '''
        #单独处理train数据集
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_wrapper.trans_indices, maxlen=self.config.get('max_history_length', 32),
            padding='pre', truncating='pre', value=-1
        ).reshape([-1])
        #iloc:位置索引
        item_indices = self.trans.iloc[trans_indices]['article_id']
        self.train_wrapper.user_indices = list(map(int, self.train_wrapper.user_indices))#map返回一个user_indices的迭代器，用list转换成列表
        data = {
            'profile': self.user_data['profile'][self.train_wrapper.user_indices],
            'context': self.trans_data['context'][trans_indices].reshape(#第327行已经预处理过了，每个人的购买记录都是等长的
                [len(self.train_wrapper), self.config.get('max_history_length', 32), -1]),
            'items': np.asarray(item_indices, np.int32).reshape([-1, self.config.get('max_history_length', 32)])#-1指将剩余元素都放着个轴上
        }#context为什么要reshape成固定形状，每个人的购买记录不一样啊？   item为什么要reshape呢，train smaples×max_history_length
        data['profile'] = np.array(data['profile'])[:,[0,1,3]]
        data['context'] = np.array(data['context'])[:,:,[2,3,4]]
        data['profile'] = data['profile'].tolist()
        data['context'] = data['context'].tolist()
        dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(2*batch_size).batch(batch_size, drop_remainder=True)
        #shuffle是防止数据过拟合的重要手段，buffer size越大混合程度越大
        return dataset

