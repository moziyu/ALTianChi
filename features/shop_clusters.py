# coding=UTF-8
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from features.week_data import get_user_pay_week_data
import params

def shop_clusters():

    #构造商家特征
    # 按周划分最近六周的shop_user样本
    week_data = get_user_pay_week_data()
    week_last_1st = week_data[1]
    week_last_2nd = week_data[2]
    week_last_3rd = week_data[3]
    week_last_4th = week_data[4]
    week_last_5th = week_data[5]
    week_last_6th = week_data[6]

    # 取倒数三周的商家流量
    predict_x = week_last_1st.join(week_last_2nd)
    predict_x = predict_x.join(week_last_3rd)

    # 将商家三周的各统计特征作为测试样本的特征
    test_mean = predict_x.mean(axis=1)
    test_median = predict_x.median(axis=1)

    predict_x['mean'] = test_mean
    predict_x['median'] = test_median

    #数据标准化
    ss_x = StandardScaler()
    predict_x = ss_x.fit_transform(predict_x)

    #根据“肘部”观察法确定聚类的个数
    K = range(1,16)
    mean_distortion = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(predict_x)
        mean_distortion.append(sum(np.min(cdist(predict_x, kmeans.cluster_centers_ , 'euclidean'),axis=1))/predict_x.shape[0])

    plt.plot(K, mean_distortion, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Average Dispersion')
    plt.title('Select label number by Elbow Method')
    plt.show()

    #根据“肘部”观察法商家应分为7类
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(predict_x)

    predict_x = pd.DataFrame(ss_x.inverse_transform(predict_x))
    cluster_label = pd.Series(kmeans.labels_)
    predict_x['cluster_label'] = cluster_label

    shop_iid = pd.DataFrame(np.arange(1,2001), columns=['iid'])
    predict_x = shop_iid.join(predict_x)
    print predict_x['cluster_label'].value_counts()

    if (not os.path.exists(params.SHOP_FEATURE_PATH)):
        os.mkdir(params.SHOP_FEATURE_PATH)
    for i in range(4):
        shops =  predict_x[predict_x['cluster_label']==i]
        print shops
        shops.to_csv(params.SHOP_FEATURE_PATH+ 'shop_clusters_' + str(i) + '.csv')

    return cluster_label


