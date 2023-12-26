# 基于多模态特征融合的个性化服装序列预测

该项目为数据挖掘课程论文的相关代码

## 数据集 
本项目的完整数据集来源于kaggle中的H&M数据集，https://www.kaggle.com/datasets/odins0n/handm-dataset-128x128/data
清洗后的商品信息、用户信息、交互信息数据集(articles3.csv,customers3.csv,transactions_train3.csv)及预处理后的图像数据(rec.csv)存放于data文件夹


## 需要的库
  - tensorflow==2.2.0
  - transformers==4.6.1
  - numpy==1.18.5
  - pandas==0.25.3
## 运行说明
- 训练
```bash
python engine.py
```

- 评估
```bash
python pinggu.py
```
