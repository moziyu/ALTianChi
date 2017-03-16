# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import params
import model_value
from features.features_to_predict_second_week import get_predict_features

def xg_regress_by_three_weeks():
    x= pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    predict_x = pd.read_csv(params.SAMPLE_PATH + 'predict_x.csv')
    columns = predict_x.columns

    #数据标准化处理
    ss_x = StandardScaler()
    ss_y = StandardScaler()
    x = ss_x.fit_transform(x)
    y = ss_y.fit_transform(y)
    predict_x = ss_x.transform(predict_x)

    #交叉验证法训练模型
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    # etr = ExtraTreesRegressor(n_estimators=1200,random_state=1,n_jobs=-1,min_samples_split=2,min_samples_leaf=2,max_depth=25,max_features=270)
    xg = XGBRegressor(learning_rate = 0.1, n_estimators = 200, max_depth = 10, min_child_weight = 2, gamma = 0.3, subsample = 0.7,
                      colsample_bytree = 0.6, scale_pos_weight = 1, seed = 27)

    off_err = []
    for train_index, test_index in kf.split(x):
        x_train, x_holdout = x[train_index], x[test_index]
        y_train, y_holdout = y[train_index], y[test_index]

        xg_test_y = []
        xg_predict_y = []
        for i in range(0, 7):
            # lr = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0])
            xg.fit(x_train, y_train[:, i])
            xg_test_y.append(xg.predict(x_holdout))
        xg_test_y = np.transpose(xg_test_y)
        err = model_value.value_mode(ss_y.inverse_transform(xg_test_y), ss_y.inverse_transform(y_holdout))
        off_err.append(err)
        print "off_line error is:", err
    # 线下误差
    err = np.mean(off_err)
    print "off_line mean error is:", err

    #预测
    xg_predict_y = []
    for j in range(0, 7):
        xg.fit(x, y[:, j])
        xg_predict_y.append(xg.predict(predict_x))
    xg_predict_y = np.transpose(xg_predict_y)
    xg_predict_y = (ss_y.inverse_transform(xg_predict_y)).astype(int)
    xg_predict_y = pd.DataFrame(xg_predict_y, columns=['day_' + str(i) for i in np.arange(1, 8)])


    # #评估模型性能
    # etr_test_y = etr.predict(test_x)
    # print 'R-squared value of etr:', etr.score(test_x, test_y)
    # print 'the Mean squared error of etr:', mean_squared_error(ss_y.inverse_transform(test_y), ss_y.inverse_transform(etr_test_y))
    # print 'the mean absolute error of etr:', mean_absolute_error(ss_y.inverse_transform(test_y),ss_y.inverse_transform(etr_test_y))

    #通过训练好的模型，判断每种特征对模型的贡献
    print np.sort(zip(xg.feature_importances_, columns),axis=0)


    #获得预测第二周的样本,并标准化
    predict_x = get_predict_features(xg_predict_y)
    predict_x = ss_x.transform(predict_x)

    xg_predict_y2 = []
    for i in range(7):
        xg.fit(x, y[:, i])
        xg_predict_y2.append(xg.predict(predict_x))
    xg_predict_y2 = np.transpose(xg_predict_y2)
    xg_predict_y2 = (ss_y.inverse_transform(xg_predict_y2)).astype(int)
    xg_predict_y2 = pd.DataFrame(xg_predict_y2, columns=['day_' + str(i) for i in np.arange(8, 15)])

    # 将etr_predict_y转为提交格式
    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join([xg_predict_y, xg_predict_y2])
    #result = pd.merge(result, result , on='iid')

    if(not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH +  'result_xg_gs_by_three_weeks.csv', index=False, header=False)

    print result.info()