from transformers import TFBertModel, BertConfig
from tensorflow.keras import layers
from evaluate import UnifiedLoss,CircleLoss
import tensorflow as tf
import os
import  pandas as pd
# 显示所有列
pd.set_option('display.max_columns',None)
# 显示所有行
pd.set_option('display.max_rows',None)



path='./bert-base-uncased/config.json'

class AttributeEmbedding(layers.Layer):
    """ 属性嵌入层 """

    def __init__(self, size, embed_dim=32, **kwargs):
        # 初始化函数，定义了类的一些属性
        super(AttributeEmbedding, self).__init__(**kwargs)#格式：spuer(classname,self).method_of_fatherclass(par)，相当于fatherclassname.method
        self.size = size# 属性种类数量
        self.embed_dim = embed_dim# 嵌入维度
        self.supports_masking = True# 是否支持遮盖

    def build(self, input_shape):
        # 构建函数，定义了需要训练的参数，一般在这里定义权重
        # 定义嵌入权重，权重形状为 (属性数量之和, 嵌入维度)
        self.embedding = self.add_weight(
            name='{}_embedding'.format(self.name),
            shape=(sum(self.size), self.embed_dim),#45×64
            initializer='normal',
            dtype=tf.float32,
            trainable=True
        )

        _cum = [0] + self.size[:-1]#去掉size最后一个元素，并在开头补充一个0
        self.cum = tf.cast(tf.cumsum(_cum), tf.int32)[tf.newaxis, :]#1×2，，，cumsum沿着某一方向累计求和
        super(AttributeEmbedding, self).build(input_shape)

    def call(self, inputs):
        #实现该层的逻辑
        indices = inputs + self.cum#计算属性在嵌入向量中的位置(input什么含义，为什么要加cum)
        embeds = tf.gather(self.embedding, indices)#就是抽取出embedding的第0维度上，indices对应的所有的值
        return tf.reduce_mean(embeds, axis=-2) #对属性的嵌入向量求平均，返回嵌入后的特征

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(AttributeEmbedding, self).get_config()
        config['size'] = self.size
        config['embed_dim'] = self.embed_dim
        return config

#
# class Image(layers.Layer):
#     """ 商品图片模型 """
#
#     def __init__(self, embed_dim=512, image_weights=None, **kwargs):
#         super(Image, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         # self.backbone = tf.keras.applications.ResNet50(
#         #     include_top=False, pooling='max',  weights=image_weights)
#
#     def call(self, img, training=None):
#         size=tf.size(img)
#         image_feature=pd.read_csv(path)
#         vector=image_feature[img]
#         # for image in images:
#         #     image = self._preprocess(image)
#         #     new_height = tf.constant(224)
#         #     new_width = tf.constant(224)
#         #     resized_image = tf.image.resize(image,[new_height,new_width])
#         #     resized_images.append(resized_image)
#         # resized_tensor = tf.stack(resized_images, axis=0)
#         # x = self.backbone(resized_tensor, training=training)
#         return vector
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.embed_dim
#
#     def get_config(self):
#         config = super(Image, self).get_config()
#         config['embed_dim'] = self.embed_dim
#         return config


class Desc(layers.Layer):
    """ 商品描述模型 """

    def __init__(self, embed_dim=32, **kwargs):
        super(Desc, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        config = BertConfig.from_json_file(path)
        self.backbone = TFBertModel(config)
        self.dense = layers.Dense(self.embed_dim,activation='relu')
        self.supports_masking=True

    def call(self, inputs, training=None):
        attention_mask = tf.not_equal(inputs, 0)#返回布尔值的张量，即如果第一个张量的值不等于第二个张量的值，则返回true;否则返回false。
        x = self.backbone(
            input_ids=inputs,
            attention_mask=attention_mask,
            training=training
        ).last_hidden_state[:, 0, :]
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embed_dim

    def get_config(self):
        config = super(Desc, self).get_config()
        config['embed_dim'] = self.embed_dim
        return config


class Item(tf.keras.Model):
    """ 商品模型 """

    def __init__(self,info_size, embed_dim=32, **kwargs):  # 可扩张参数是字典形式
        image_weights=kwargs.pop('image_weights',None)
        bert_path = kwargs.pop('bert_path', None)
        super(Item, self).__init__(**kwargs)
        # self.image_model=Image(
        #     embed_dim,image_weights=image_weights,name='Image'
        # )
        self.desc_model = Desc(embed_dim, name='Desc')
        self.info_model = AttributeEmbedding(info_size, embed_dim, name='Info')
        self.ln = layers.BatchNormalization()
        self.a = layers.Activation('relu')
        self.dense = layers.Dense(embed_dim,activation='relu')

    def call(self, inputs,training=None ):
        # img_embed = self.image_model(inputs['image'])
        desc_embed = self.desc_model(inputs['desc'],training=training)
        info_embed = self.info_model(inputs['info'])
        #inputs['image']=tf.reshape(inputs['image'],(-1,64))
        #x = layers.Add()([inputs['image'], desc_embed,info_embed])
        #x = layers.Add()([desc_embed, info_embed])
        x = layers.concatenate( [desc_embed,info_embed],axis=1)
        x = self.ln(x)
        x = self.a(x)
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.desc_model.embed_dim


class User(tf.keras.Model):
    """ 用户模型 """
    def __init__(self, profile_size, context_size, embed_dim=32, **kwargs):
        dropout = kwargs.pop('dropout', 0.0)
        recurrent_dropout = kwargs.pop('recurrent_dropout', dropout)
        super(User, self).__init__(**kwargs)
        self.profile_model = tf.keras.Sequential([AttributeEmbedding(profile_size, embed_dim)], name='Profile')#将用户向量重新嵌入到新的维度
        self.trans_model = layers.TimeDistributed(AttributeEmbedding(context_size, embed_dim), name='Context')#TimeDistributed的作用是将AttributeEmbedding应用到每个输入上，例如输入的一个样本的特征为（10，6），则TimeDistributed将AttributeEmbedding分别应用到这个10个向量上，生成新的特征向量例如（10，9）
        self.ln = layers.BatchNormalization()
        self.a = layers.Activation('relu')
        # self.masking = layers.Masking(0.0)
        self.history_model = layers.GRU(
            embed_dim, recurrent_activation='hard_sigmoid',
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=True, return_state=True
        )
 	 
 

    def call(self, inputs, training=None):
        #call方法中实现网络的前向传播结构(即在这个方法中定义网络结构)
        # initial_state = inputs.get('initial_state')
        # if initial_state is None:

        user = self.profile_model(inputs['profile'])#重组profile特征
        context = self.trans_model(inputs['context'])#重组context购买记录特征
        # items = self.masking(inputs['items'])
        items=inputs['items']
        user = tf.repeat(user[:, tf.newaxis, :], repeats=20, axis=1)
        #x = layers.Add()([context, items,user])#把购买记录：userid;itemid;itemvector拼接在一起形成新的向量x
        x = layers.concatenate([context, items,user],axis=2)
        print(x)
        x=self.ln(x)
        x=self.a(x)
        y, h = self.history_model(x, training=training)#为什么要用profile来初始化GRU呢？优点是什么
        return y, h

    def infer_initial_state(self, profile, batch_size=32, **kwargs):
        return self.profile_model.predict(
            profile, batch_size=batch_size, **kwargs)


def build_model(config):
    #定义商品模型
    item_model = Item(
        info_size=config['info_size'],
        embed_dim=config.get('embed_dim', 32),
        image_weights=config['image_weights'],
        name='Item'
    )
    h, w = config.get('image_height', 256), config.get('image_width', 256)
    item_dummy_inputs = {
        'image': layers.Input(shape=(h, w)),
        'desc': layers.Input(shape=(config['max_desc_length'],), dtype=tf.int32),
        'info': layers.Input(shape=(len(config['info_size']),), dtype=tf.int32)
    }
    item_model(item_dummy_inputs)
    item_model.summary()

    #定义用户模型
    user_model = User(
        config['profile_size'], config['context_size'],
        embed_dim=config.get('embed_dim', 32),
        name='User'
    )


    train_dummy_inputs = {
        'items': layers.Input(shape=(1, config.get('embed_dim', 32)), dtype=tf.float32),
        'desc': layers.Input(shape=(1, config['max_desc_length']), dtype=tf.int32),
        'info': layers.Input(shape=(1, len(config['info_size'])), dtype=tf.int32),
        'profile': layers.Input(shape=(len(config['profile_size']),), dtype=tf.int32),
        'context': layers.Input(shape=(1, len(config['context_size'])), dtype=tf.int32)
    }
    user_model(train_dummy_inputs)
    user_model.summary()
    return item_model,user_model


class RecModel(tf.keras.Model):
    """ Recommendation Model for Training """

    def __init__(self, config, item_model, user_model, item_data, **kwargs):
        super(RecModel, self).__init__(**kwargs)
        self.config = config
        self.item_model = item_model
        self.user_model = user_model
        # Cache item data to accelarate
        with tf.device(item_model.trainable_weights[0].device):
            self.item_data = {
                'info': tf.identity(item_data['info']),
                'desc': tf.identity(item_data['desc']),
                'image': tf.identity(item_data['image'])
            }

    def compile(self, optimizer, margin=0.0, gamma=1.0):
        super(RecModel, self).compile(optimizer=optimizer)
        self.loss_fn = UnifiedLoss(
            margin=margin, gamma=gamma,
            reduction=tf.keras.losses.Reduction.NONE
        )
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape(persistent=True) as tape:  # persistent: 布尔值，用来指定新创建的gradient tape是否是可持续性的。默认是False，意味着只能够调用一次gradient（）函数。
            batch_size = tf.shape(inputs['items'])[0]
            seq_length = tf.shape(inputs['items'])[1]
            # compute item vectors
            item_indices = tf.reshape(inputs['items'], [-1])  # 拉平成一维,每次处理8×5个样本
            pad_mask = tf.not_equal(item_indices, -1)
            item_vectors = self.item_model(
                {
                    'info': tf.gather(self.item_data['info'], item_indices),
                    'desc': tf.gather(self.item_data['desc'], item_indices),
                    'image': tf.gather(self.item_data['image'], item_indices)
                },
                training=True
            )#(40,64)
            item_vectors *= tf.expand_dims(tf.cast(pad_mask, tf.float32), -1)  #expand_dims增加一个维度， 将无效的商品向量置为0
            item_vectors = tf.reshape(item_vectors,[batch_size, seq_length, -1])  # 将商品向量恢复为(batch_size, seq_length, -1)的形状
            # compute user vectors
            state_seq, _ = self.user_model(
                {
                    'profile': inputs['profile'],
                    'context': inputs['context'],
                    'items': item_vectors
                },
                training=True
            )  # 根据用户历史行为和当前商品向量计算用户向量
            batch_idx = tf.range(0, batch_size)
            length_idx = tf.range(0, seq_length)
            a = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
            b = length_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
            c = batch_idx[tf.newaxis, tf.newaxis, :, tf.newaxis]
            d = length_idx[tf.newaxis, tf.newaxis, tf.newaxis, :]
            # mask掉历史商品和预测长度之外的商品
            # mask history items and items out of prediction length
            prd_mask = tf.logical_and(
                tf.equal(a, c),
                tf.logical_or(
                    tf.greater_equal(b, d), tf.greater(d - b, self.config['predict_length']))
            )  # 根据序列索引计算出哪些位置的商品不需要被预测
            prd_mask = tf.reshape(prd_mask, [batch_size, seq_length,-1])  # (batch, len, batch * len)
            prd_mask = tf.logical_not(prd_mask)  # 取反，得到哪些位置的商品需要被预测
            pad_mask = pad_mask[tf.newaxis, tf.newaxis, :]  # 将pad_mask的形状变为(1，1，batch * len)，方便后面的运算
            mask = tf.logical_and(pad_mask, prd_mask)
            # compute logits,一般是softmax之前的一层
            item_vectors = tf.reshape(item_vectors, [-1, self.config['embed_dim']])  # (batch * len, dim)
            #让每一个样本与每次处理的所有样本进行对比，这里一共有8批样本，每一批中有5个样本。所以就是让这5个样本分别与40个样本进对比，这种操作进行8次，即40个样本之间互相比较，来算样本之间的相似度。
            logits = tf.matmul(state_seq, item_vectors, transpose_b=True)  # (batch, len, batch * len)
            # compute labels
            labels = tf.tile(tf.equal(a, c), [1, seq_length, 1, seq_length])#在某一维度上进行复制，第二个list参数数组的值代表复制的次数
            labels = tf.cast(tf.reshape(labels, [batch_size, seq_length, -1]), tf.float32)#会形成5列连在一起的形状，这是因为这5个商品都是一个用户买的，属于同一类
            labels = tf.cast(tf.where(mask, labels, -1), labels.dtype)#如果需要mask则将label置为-1，使其无效
            # Loss= UnifiedLoss(margin=0.3,gamma=1)
            # Loss=CircleLoss(margin=0,gamma=1)
            # loss = Loss.call(labels, logits)
            loss = self.loss_fn(labels, logits)
        variables = self.item_model.trainable_weights + self.user_model.trainable_weights
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {'loss': loss}


    def save_weights(self, filepath, **kwargs):
        self.item_model.save_weights(os.path.join(filepath, 'item.h5'), **kwargs)
        self.user_model.save_weights(os.path.join(filepath, 'user.h5'), **kwargs)


class RecInfer(tf.keras.Model):
    def __init__(self, user_model, skip_used_items=False, **kwargs):
        top_k = kwargs.pop('top_k', 10)
        max_history_length = kwargs.pop('max_history_length', 32)
        profile_dim = kwargs.pop('profile_dim')
        context_dim = kwargs.pop('context_dim')
        num_items = kwargs.pop('num_items')
        embed_dim = kwargs.pop('embed_dim')
        super(RecInfer, self).__init__(**kwargs)
        self.skip_used_items = skip_used_items
        self.top_k = top_k
        self.user_model = user_model
        self.item_vectors = self.add_weight(name='item_vectors',
                                            shape=(num_items, embed_dim),
                                            dtype=tf.float32,
                                            initializer='zeros',
                                            trainable=False)

        dummy_inputs = {
            'profile': layers.Input(shape=(profile_dim,), dtype=tf.int32),
            'context': layers.Input(shape=(max_history_length, context_dim), dtype=tf.int32),
            'item_indices': layers.Input(shape=(max_history_length,), dtype=tf.int32)
        }
        self(dummy_inputs)

    def set_item_vectors(self, item_vectors):
        self.item_vectors.assign(item_vectors)

    def call(self, inputs):
        batch_size = tf.shape(inputs['item_indices'])[0]
        seq_length = tf.shape(inputs['item_indices'])[1]
        item_indices = tf.reshape(inputs.pop('item_indices'), [-1])
        inputs['items'] = tf.reshape(
            tf.gather(self.item_vectors, item_indices),
            [batch_size, seq_length, -1]
        )
        _, user_vector = self.user_model(inputs, training=False)
        score = tf.matmul(user_vector, self.item_vectors, transpose_b=True)
        # if self.skip_used_items:
        #     # mask used_items
        #     used_items = tf.reshape(item_indices, [batch_size, seq_length])
        #     item_size = tf.shape(self.item_vectors)[0]
        #     used_items = tf.reshape(used_items, [-1])
        #     mask = tf.one_hot(used_items, depth=item_size, dtype=tf.int8)
        #     mask = tf.reshape(mask, [batch_size, -1, item_size])
        #     mask = tf.reduce_any(tf.not_equal(mask, 0), axis=1)
        #     score -= 1e5*tf.cast(mask, score.dtype)
        recommend = tf.argsort(score, direction='DESCENDING')[:, :20]#将返回每个样本的排序后的索引。
        return recommend
