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
from models.model_value import value_mode


def ert_gs_regress():
    train_x= pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    train_y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    validation_x= pd.read_csv(params.SAMPLE_PATH + 'valution_x.csv')
    validation_y = pd.read_csv(params.SAMPLE_PATH + 'valution_y.csv')
    predict_x = pd.read_csv(params.SAMPLE_PATH + 'predict_x.csv')
    columns = predict_x.columns
    print train_x

    #数据标准化处理
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    train_x = ss_x.fit_transform(train_x)
    train_y = ss_y.fit_transform(train_y)
    validation_x = ss_x.transform(validation_x)
    validation_y = ss_y.transform(validation_y)
    predict_x = ss_x.transform(predict_x)

    #etr = ExtraTreesRegressor()
    #etr = ExtraTreesRegressor(n_estimators=1200,random_state=1,n_jobs=-1,min_samples_split=2,min_samples_leaf=2,max_depth=24)
    parameters = {'n_estimators':[400,800,1200,1600,2000], 'min_samples_split':[2,3,4], 'max_depth':[14,15,16], 'min_samples_leaf':[2,3,4]}
    etr = ExtraTreesRegressor()
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    loss = make_scorer(value_mode, greater_is_better=False)
    gs = GridSearchCV(etr, param_grid=parameters, cv=kf, refit=True, scoring=loss, n_jobs=-1)
    gs.fit(train_x, train_y)

    # 通过训练好的模型进行预测
    gs_validation_y = gs.predict(validation_x)
    gs_predict_y = gs.predict(predict_x)
    gs_predict_y = (ss_y.inverse_transform(gs_predict_y)).astype(int)
    etr_predict_y = pd.DataFrame(gs_predict_y)
    print "off_line error is:", value_mode(ss_y.inverse_transform(gs_validation_y),
                                                       ss_y.inverse_transform(validation_y))  # 线下误差
    # 将etr_predict_y转为提交格式
    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join(etr_predict_y)
    result = pd.merge(result, result, on='iid')

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_etr_gs_by_three_weeks.csv', index=False, header=False)

    print gs.best_params_
    print gs.best_score_



