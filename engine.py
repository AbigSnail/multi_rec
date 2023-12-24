from transformers import BertConfig, BertTokenizer
import tensorflow as tf
import numpy as np
import pandas as pd
from model_GRU import RecModel, build_model, RecInfer
from data import RecData


import os

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

bertpath = './bert-base-uncased/'
trans_data_path = './data/transactions_train3.csv'
user_data_path = './data/customers3.csv'
item_data_path = './articles3.csv'
save_path = './weight/'


class RecEngine:
    def __init__(self, config: dict):
        self.config = config
        print(config)
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.item_model, self.user_model = build_model(self.config)

    @classmethod
    def from_pretrained(cls, model_path: str):
        # model_config = BertConfig.from_pretrained(bertpath,output_attentions=True)

        config = {

            'max_history_length': 20,
            'predict_length': 10,
            'max_desc_length': 10,

            'info_size': [50, 239, 21, 30, 5, 10, 23177, 16],
            'profile_size': [67, 2, 9663],  # profile由4个属性构成,这里使其对要处理的属性进行选择，比如说这里有4个9，那么只选取了前4个属性进行预测
            'context_size': [333, 2, 734],
            'embed_dim': 64,  # GRU的输出是64维度的
            'bert_path': bertpath,
            'image_weights': 'imagenet',
            'image_height': 8,
            'image_width': 8
        }
        engine = cls(config)  # cls在python中表示类本身，self为类的一个实例。
        engine.item_model.load_weights(os.path.join(model_path, 'item.h5'))
        engine.user_model.load_weights(os.path.join(model_path, 'user.h5'))
        return engine

    def train(self, data: RecData,
              test_users: list = None,
              save_path: str = './model',
              **kwargs):
        strategy = tf.distribute.MirroredStrategy()
        print("REPLICAS: ", strategy.num_replicas_in_sync)

        os.makedirs(save_path, exist_ok=True)
        batch_size = kwargs.get('batch_size', 64)
        data.prepare_features(self.tokenizer)
        data.prepare_train(test_users)
        dataset = data.train_dataset(batch_size)
     
        model_config = BertConfig.from_pretrained(self.config.get('bert_path', 'bert-base-uncased'))
        model_config.update(self.config)
        model_config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # data.save_feature_dict(save_path)
        # Compile model with optimizer
        
        
        with strategy.scope():
          
        
          self.item_model, self.user_model = build_model(self.config)
          rec_model = RecModel(self.config,
                             self.item_model,
                             self.user_model,
                             data.item_data)
          total_steps = kwargs.get('epochs', 100) * len(data.train_wrapper) // kwargs.get('batch_size', 64)
          optimizer = tf.keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999)
  
          # optimizer = AdamWarmup(
          #     warmup_steps=int(total_steps * kwargs.get('warmup_proportion', 0.1)),
          #     decay_steps=total_steps - int(total_steps * kwargs.get('warmup_proportion', 0.1)),
          #     initial_learning_rate=kwargs.get('learning_rate', 1e-4),
          #     lr_multiply=kwargs.get('lr_multiply')
          # )
  
          rec_model.compile(
              optimizer=optimizer,
              margin=0.25,
              gamma=1,
          )
        
        # 创建TensorBoard回调
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

          # 训练模型并添加TensorBoard回调
          # with tf.device(self.item_model.trainable_weights[0].device):
          #     checkpoint = Checkpoint(
          #         save_path, data, self.config,
          #         batch_size=kwargs.get('infer_batch_size',64),
          #         skip_used_items=kwargs.get('skip_used_items', False),
          #         verbose=1
          #     )
        rec_model.fit(
              dataset,
              batch_size=batch_size,
              epochs=60,
              callbacks=[tensorboard_callback],
          )
         
        rec_model.save_weights(save_path)

    def infer(self, data: RecData,
              batch_size: int = 64,
              skip_used_items=False,
              top_k=10,
              verbose=1):
        data.prepare_features(self.tokenizer)
        infer_wrapper = data.infer_wrapper
        trans_indices = tf.keras.preprocessing.sequence.pad_sequences(
            infer_wrapper.trans_indices, maxlen=self.config.get('max_history_length', 50), value=-1,truncating='post'
        ).reshape([-1])
        # 这可以看用户的id
        profile = np.array(data.user_data['profile'][infer_wrapper.user_indices])
        context = np.array(data.trans_data['context'][trans_indices].reshape(
            [len(profile), self.config.get('max_history_length', 50), -1]))
        item_indices = np.asarray(  # trans都是映射之后的结果
            data.trans.iloc[trans_indices]['article_id'], np.int32
        ).reshape([len(profile), -1])

        profile = np.array(data.user_data['profile'][infer_wrapper.user_indices])[:, [0, 1, 3]]
        context = np.array(data.trans_data['context'][trans_indices].reshape(
            [len(profile), self.config.get('max_history_length', 50), -1]))[:, :, [2, 3, 4]]
        item_indices = np.asarray(  # trans都是映射之后的结果
            data.trans.iloc[trans_indices]['article_id'], np.int32
        ).reshape([len(profile), -1])

        infer_model = RecInfer(self.user_model,
                               skip_used_items=skip_used_items,
                               max_history_length=self.config['max_history_length'],
                               profile_dim=len(self.config['profile_size']),
                               context_dim=len(self.config['context_size']),
                               num_items=len(data.items),
                               embed_dim=self.config['embed_dim'],
                               top_k=top_k)
        item_vectors = self.item_model.predict(data.item_data,  # 获取所有商品的向量
                                               batch_size=1,
                                               verbose=verbose
                                               )

        infer_model.set_item_vectors(item_vectors)
        infer_inputs = {'profile': profile, 'context': context, 'item_indices': item_indices}
        predictions = infer_model.predict(infer_inputs, batch_size=1, verbose=verbose)  # verbose为1显示进度条，否则不显示

        # for i in range(tf.size(predictions)[0]):
        #     hit_ratio_list.append(hit_ratio(infer_wrapper.trans_indices[i],predictions[i,:]))
        # print(hit_ratio_list)
        return predictions


if __name__ == '__main__':
 
    config = {
        'batch_size': 32,
        'max_history_length': 20,
        'predict_length': 10,
        'max_desc_length': 10,
        # 'info_size': [48, 202, 21,29,5,10,5568,13],
        # 'profile_size': [73, 2],  # profile由4个属性构成,这里使其对要处理的属性进行选择，比如说这里有4个9，那么只选取了前4个属性进行预测
        # 'context_size': [1458, 2, 27],
        'info_size': [50, 239, 21, 30, 5, 10, 23177, 16],
        'profile_size': [67, 2, 9663],  # profile由4个属性构成,这里使其对要处理的属性进行选择，比如说这里有4个9，那么只选取了前4个属性进行预测
        'context_size': [333, 2, 734],
        'embed_dim': 64,  # GRU的输出是64维度的
        'bert_path': bertpath,
        'image_weights': 'imagenet',
        'image_height': 8,
        'image_width': 8
    }
    rec = RecEngine(config)
    df = pd.read_csv(trans_data_path)
    af = pd.read_csv(item_data_path)
    cf = pd.read_csv(user_data_path)
    data = RecData(af, cf, df, config)
    test_users = []

    rec.train(data, test_users=test_users, save_path=save_path, batch_size=32)
    engine=rec.from_pretrained('./weight/')
    rec.item_model=engine.item_model
    rec.user_model=engine.user_model
    print('开始预测')
    result=rec.infer(data,64)
    df=pd.DataFrame(result)
    df.to_csv('outdata.csv',index=False,header=False)
    print(result)
    print(result.shape)
