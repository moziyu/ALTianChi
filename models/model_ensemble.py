# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn import linear_model
import os
import params
import model_ensemble_class
import model_value

# Stacking
def stack_model():
    x = pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    predict_x = pd.read_csv(params.SAMPLE_PATH + 'predict_x.csv')

    # 数据标准化处理
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    x = ss_x.fit_transform(x)
    y = ss_y.fit_transform(y)
    predict_x = ss_x.transform(predict_x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)

    base_models = [ExtraTreesRegressor(n_estimators=600,random_state=1,n_jobs=-1,min_samples_split=2,min_samples_leaf=2,max_depth=24),
            RandomForestRegressor(n_estimators=1200, random_state=1, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,max_depth=25),
            XGBRegressor(max_depth=6, n_estimators=500),
            linear_model.LassoCV(alphas=[0.02,0.05,0.1,0.5,1.0])]

    alphas = [1.8]
    for n in alphas:
        stacker = linear_model.Ridge(alpha=n)
        #预训练模型，评估线下误差
        ens_test = []
        for m in range(7):
            ensemble_model = model_ensemble_class.Ensemble(n_folds=5, base_models=base_models, stacker=stacker)
            ens_test.append(ensemble_model.fit_predict(train_x, train_y[:, m], test_x))
        ens_test = np.transpose(ens_test)
        print "off_line error is:", model_value.value_mode(ss_y.inverse_transform(ens_test),
                                                           ss_y.inverse_transform(test_y))  # 线下误差

    #预测第一周流量
    ens_predict = []
    for m in range(7):
        ensemble_model = model_ensemble_class.Ensemble(n_folds=5, base_models=base_models, stacker=stacker)
        ens_predict.append(ensemble_model.fit_predict(x, y[:, m], predict_x))

    ens_predict = np.transpose(ens_predict)
    ens_predict = (ss_y.inverse_transform(ens_predict)).astype(int)
    ens_predict = pd.DataFrame(ens_predict)

    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join(ens_predict)
    result = pd.merge(result, result, on='iid')

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result.iloc[i,j] < 0:
                result.iloc[i,j] = 0

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_ens_by_two_weeks.csv', index=False, header=False)

    print result

    # clf = RandomForestRegressor(n_estimators=1000, max_depth=8)
    # clf.fit(train_stack, train_y)
    # predict = clf.predict(test_stack)