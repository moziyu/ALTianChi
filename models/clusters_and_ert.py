# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import os
import model_value
import params
from features.shop_clusters import shop_clusters

def ert(train_x, train_y, predict_x):
    # 训练模型
    etr = ExtraTreesRegressor(n_estimators=800, random_state=1, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                              max_depth=27)
    etr.fit(train_x, train_y)
    etr_predict_y = etr.predict(predict_x)
    return etr_predict_y

# 获取商家聚类数据并构造样本
def clustes_ert():

    train_x= pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    train_y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    validation_x= pd.read_csv(params.SAMPLE_PATH + 'validation_x.csv')
    validation_y = pd.read_csv(params.SAMPLE_PATH + 'validation_y.csv')
    predict_x = pd.read_csv(params.SAMPLE_PATH + 'predict_x.csv')

    #数据标准化处理
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    train_x = ss_x.fit_transform(train_x)
    train_y = ss_y.fit_transform(train_y)
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    validation_x = pd.DataFrame(ss_x.transform(validation_x))
    validation_y = pd.DataFrame(ss_y.transform(validation_y))
    predict_x = pd.DataFrame(ss_x.transform(predict_x))

    clusters_label = shop_clusters()

    train_x['clusters_label'] = clusters_label
    train_y['clusters_label'] = clusters_label
    validation_x['clusters_label'] = clusters_label
    validation_y['clusters_label'] = clusters_label
    predict_x['clusters_label'] = clusters_label
    validation_x['iid'] = pd.Series(np.arange(1, 2001))
    predict_x['iid'] = pd.Series(np.arange(1, 2001))

    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)
    result_validation = []
    result_predict = []
    for i in range(4):
        cluster_x = train_x[train_x['clusters_label'] == i]
        cluster_y = train_y[train_y['clusters_label']==i]
        cluster_x = cluster_x.drop('clusters_label', axis=1)
        cluster_y = cluster_y.drop('clusters_label', axis=1)

        x_validation = validation_x[validation_x['clusters_label'] == i]
        x_validation_iid = x_validation['iid']
        x_validation = x_validation.drop(['clusters_label','iid'], axis=1)
        y_validation = validation_y[validation_y['clusters_label'] == i]
        y_validation = y_validation.drop('clusters_label', axis=1)

        x_predict = predict_x[predict_x['clusters_label'] == i]
        x_predict_iid = x_predict['iid']
        x_predict = x_predict.drop(['clusters_label','iid'], axis=1)

        y_validation_predict = ert(cluster_x.values, cluster_y.values, x_validation.values)
        y_validation_predict = pd.DataFrame(y_validation_predict)

        y_predict = ert(x_validation.values, y_validation.values, x_predict.values)
        y_predict = pd.DataFrame(y_predict)

        y_validation_predict['iid'] = np.array(x_validation_iid)
        y_predict['iid'] = np.array(x_predict_iid)

        result_validation.append(y_validation_predict)
        result_predict.append(y_predict)

    result_validation = pd.concat(result_validation)
    result_validation.index = np.arange(result_validation.shape[0])
    # 按照iid降序排列
    result_validation = result_validation.sort_values(by='iid',ascending=True)
    result_validation = result_validation.drop('iid', axis=1)
    result_validation = (ss_y.inverse_transform(result_validation)).astype(int)
    # 评估模型性能
    validation_y = validation_y.drop('clusters_label', axis=1)
    print "off_line error is:", model_value.value_mode(result_validation, validation_y)  # 线下误差

    result_predict = pd.concat(result_predict)
    result_predict.index = np.arange(result_predict.shape[0])
    # 按照iid降序排列
    result_predict = result_predict.sort_values(by='iid',ascending=True)
    result_predict = result_predict.drop('iid', axis=1)
    result_predict = pd.DataFrame((ss_y.inverse_transform(result_predict)).astype(int))


    predict = pd.DataFrame(np.arange(1, result_predict.shape[0]+1), columns=['iid'])
    predict = predict.join(result_predict)
    predict = pd.merge(predict, predict, on='iid')

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    predict.to_csv(params.OUTPUT_PATH + 'result_clusters_and_ert_by_three_weeks.csv', index=False, header=False)

    print predict



