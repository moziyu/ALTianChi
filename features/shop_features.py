# coding=UTF-8

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cPickle
import params
import os

def get_shop_feature():
    data = pd.read_csv(params.DATA_PATH + 'shop_info.txt', header=None)
    data.columns = ['iid', 'city', 'location', 'per_pay', 'score', 'comment', 'level', 'cate1', 'cate2','cate3']

    # #按照店铺的品类等级分类
    # data_recreation = data[data['cate1']=='休闲娱乐']
    # data_health = data[data['cate1']=='医疗健康']
    # data_hair = data[data['cate1']=='美发/美容/美甲']
    # data_shoping = data[data['cate1']=='购物']
    # data_market = data[data['cate1']=='超市便利店']
    # data_food = data[data['cate1']=='美食']

    # print data.loc[:, [4,7]].groupby([7]).mean()
    # 数据填充，只有超市便利店有数据缺失
    data.loc[:,'score'].fillna(3, inplace = True)
    data.loc[:,'comment'].fillna(1.0, inplace = True)

    #将城市特征转化为OneHotEncoder编码
    #print  data['city'].value_counts()
    x = data['city'].value_counts()
    data['city'] = data['city'].replace(x.index, range(1,len(x)+1))
    data['city'] = data['city'].replace(range(12, len(x) + 1), 100*np.ones(len(x) - 11))
    city_ohe = pd.get_dummies(data['city'])
    city_ohe.columns = ['city_ohe_' + str(i) for i in range(city_ohe.shape[1])]
    print '####city_ohe',city_ohe

    # 将一级品类特征转化为OneHotEncoder编码
    print data['cate1'].value_counts()
    x = data['cate1'].value_counts()
    data['cate1'] = data['cate1'].replace(x.index, range(1, len(x) + 1))
    data['cate1'] = data['cate1'].replace(range(3, len(x) + 1), 100 * np.ones(len(x) - 2))
    cate1_ohe = pd.get_dummies(data['cate1'])
    cate1_ohe.columns = ['cate1_ohe_' + str(i) for i in range(cate1_ohe.shape[1])]
    print '####cate1_ohe', cate1_ohe

    # 将二级品类特征转化为OneHotEncoder编码
    print data['cate2'].value_counts()
    x = data['cate2'].value_counts()
    data['cate2'] = data['cate2'].replace(x.index, range(1,len(x)+1))
    data['cate2'] = data['cate2'].replace(range(11, len(x) + 1), 100 * np.ones(len(x) - 10))
    cate2_ohe = pd.get_dummies(data['cate2'])
    cate2_ohe.columns = ['cate2_ohe_' + str(i) for i in range(cate2_ohe.shape[1])]
    print '####cate2_ohe', cate2_ohe

    # 将三级品类特征转化为OneHotEncoder编码
    print data['cate3'].value_counts()
    x = data['cate3'].value_counts()
    data['cate3'] = data['cate3'].replace(x.index, range(1, len(x) + 1))
    data['cate3'] = data['cate3'].replace(range(10, len(x) + 1), 100 * np.ones(len(x) - 9))
    cate3_ohe = pd.get_dummies(data['cate3'])
    cate3_ohe.columns = ['cate3_ohe_' + str(i) for i in range(cate3_ohe.shape[1])]
    print '####cate3_ohe', cate3_ohe

    # 将评分特征转化为OneHotEncoder编码
    score_ohe = pd.get_dummies(data['score'])
    score_ohe.columns = ['score_ohe_' + str(i) for i in range(score_ohe.shape[1])]
    print '####score_ohe', score_ohe

    # 门店等级转化为OneHotEncoder编码
    level_ohe = pd.get_dummies(data['level'])
    level_ohe.columns = ['level_ohe_' + str(i) for i in range(level_ohe.shape[1])]
    print '####level_ohe', level_ohe

    # 用PCA对生成的所有One_hot向量进行降维
    oht_features = city_ohe.join([cate1_ohe, cate2_ohe, cate3_ohe, score_ohe, level_ohe])
    pca = PCA(n_components=10)
    oht_features = pca.fit_transform(oht_features)
    oht_features = pd.DataFrame(oht_features, columns=['shop_features_' + str(i) for i in np.arange(10)])

    features = pd.DataFrame(np.arange(1,2001), columns=['iid'])
    features = features.join(oht_features)
    features['per_pay'] = data.loc[:, 'per_pay'].astype(float)
    features['comment'] = data.loc[:, 'comment']

    print features

    if(not os.path.exists(params.SHOP_FEATURE_PATH)):
        os.mkdir(params.SHOP_FEATURE_PATH)
    f = open(params.SHOP_FEATURE_PATH + "shop_features.pkl", 'wb')
    cPickle.dump(features, f, -1)
    f.close()

    return features