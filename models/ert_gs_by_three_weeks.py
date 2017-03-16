# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import params
import model_value


def ert_gs_by_three_weeks():
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
    parameters = {'n_estimators':[500,600,700,800], 'min_samples_split':[2,3,4], 'max_depth':[13,14,15,16,17], 'min_samples_leaf':[2,3,4]}
    etr = ExtraTreesRegressor()
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    loss = make_scorer(model_value.value_mode, greater_is_better=False)
    gs = GridSearchCV(etr, param_grid=parameters, cv=kf, refit=True, scoring=loss, n_jobs=-1)
    gs.fit(train_x, train_y)

    # 通过训练好的模型进行预测
    gs_test_y = gs.predict(test_x)
    gs_predict_y = gs.predict(predict_x)
    gs_predict_y = (ss_y.inverse_transform(gs_predict_y)).astype(int)
    etr_predict_y = pd.DataFrame(gs_predict_y)
    print "off_line error is:", model_value.value_mode(ss_y.inverse_transform(gs_test_y),
                                                       ss_y.inverse_transform(test_y))  # 线下误差
    # 将etr_predict_y转为提交格式
    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join(etr_predict_y)
    result = pd.merge(result, result, on='iid')

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_etr_by_three_weeks.csv', index=False, header=False)
    print gs.best_params_
    print gs.best_score_
    print gs.score(test_x, test_y)



