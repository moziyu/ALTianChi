# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import params
import model_value


def xg_gs_by_three_weeks():
    x = pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    predict_x = pd.read_csv(params.SAMPLE_PATH + 'predict_x.csv')
    columns = predict_x.columns

    #数据标准化处理
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    x = ss_x.fit_transform(x)
    y = ss_y.fit_transform(y)
    predict_x = ss_x.transform(predict_x)
    # 训练模型(后四周作为训练集，采样其中的25%用于测试)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=33)

    #etr = ExtraTreesRegressor()
    #etr = ExtraTreesRegressor(n_estimators=1200,random_state=1,n_jobs=-1,min_samples_split=2,min_samples_leaf=2,max_depth=24)
    # params = [learning_rate = 0.1, n_estimators = 1000, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8,
    # colsample_bytree = 0.8, objective = 'binary:logistic', nthread = 4, scale_pos_weight = 1, seed = 27]
    parameters = {'max_depth':[10],
                  'min_child_weight':[2],
                  'gamma':[0.3],
                  'subsample':[i / 10.0 for i in range(6, 10)],
                  'colsample_bytree':[i / 10.0 for i in range(6, 10)],
                  'scale_pos_weight': [1],
                  'learning_rate': [0.1],
                  'n_estimators':[200],
                  'seed':[27]}
    xg = XGBRegressor()
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    loss = make_scorer(model_value.value_mode, greater_is_better=False)
    gs = GridSearchCV(xg, param_grid=parameters, cv=kf, refit=True, scoring=loss, iid=False, n_jobs=-1)


    # 通过训练好的模型进行误差评估
    gs_test_y =[]
    for i in range(7):
        gs.fit(train_x, train_y[:, i])
        md = gs.predict(test_x)
        gs_test_y.append(md)
        print gs.best_params_
        print gs.best_score_
        print gs.score(test_x, md)
    gs_test_y = np.transpose(gs_test_y)
    gs_test_y = (ss_y.inverse_transform(gs_test_y))
    print "off_line error is:", model_value.value_mode(ss_y.inverse_transform(gs_test_y),
                                                       ss_y.inverse_transform(test_y))  # 线下误差
    gs_test_y = pd.DataFrame(gs_test_y, columns=['day_' + str(i) for i in np.arange(1, 8)])


    # #可视化观察结果
    # data = pd.read_pickle(params.DATA_PATH + 'user_pay_last_three_months.pkl')
    # # 要查看的时期(可查看2016/08/01-2.16/10/31)
    #
    # data_colunms = range(86, 93)
    # gs_test_y.columns = np.arange(1,8)
    # # 要查看的商家
    # shop_iid = np.arange(1, 100)
    # for iid in shop_iid:
    #     data_plot = data.iloc[iid - 1, data_colunms]
    #     result1_plot = gs_test_y.iloc[iid - 1, :]
    #
    #     # data_plot.plot()
    #     plt.plot(range(1, 8), data_plot)
    #     result1_plot.plot()
    #
    #     print data_plot
    #     print result1_plot
    #
    #     plt.show()

    # 通过训练好的模型进行预测
    gs_predict_y = []
    for i in range(7):
        gs.fit(x, y[:, i])
        md = gs.predict(predict_x)
        gs_predict_y.append(md)
    gs_predict_y = np.transpose(gs_predict_y)
    gs_predict_y = (ss_y.inverse_transform(gs_predict_y))
    gs_predict_y = pd.DataFrame(gs_predict_y)

    result = pd.DataFrame(np.arange(1, 2001), columns=['iid']).astype(int)
    result = result.join(gs_predict_y)
    result = pd.merge(result, result, on='iid')
    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_xg_by_three_weeks.csv', index=False, header=False)
    print result




