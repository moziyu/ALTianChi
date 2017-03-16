# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import params
import model_value
from features.features_to_predict_second_week import get_predict_features

def rf_regress_by_three_weeks():
    train_x= pd.read_csv(params.SAMPLE_PATH + 'train_x.csv')
    train_y = pd.read_csv(params.SAMPLE_PATH + 'train_y.csv')
    validation_x= pd.read_csv(params.SAMPLE_PATH + 'validation_x.csv')
    validation_y = pd.read_csv(params.SAMPLE_PATH + 'validation_y.csv')
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

    #交叉验证法训练模型
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    # etr = ExtraTreesRegressor(n_estimators=1200,random_state=1,n_jobs=-1,min_samples_split=2,min_samples_leaf=2,max_depth=25,max_features=270)
    etr = RandomForestRegressor(n_estimators=1200, random_state=1, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                              max_depth=15, max_features=15)

    x = train_x
    y = train_y
    off_err = []
    for train_index, test_index in kf.split(x):
        x_train, x_holdout = x[train_index], x[test_index]
        y_train, y_holdout = y[train_index], y[test_index]
        etr.fit(x_train, y_train)
        etr_test_y = etr.predict(x_holdout)
        err = model_value.value_mode(ss_y.inverse_transform(etr_test_y), ss_y.inverse_transform(y_holdout))
        off_err.append(err)
        print "off_line error is:", err
    # 线下误差
    err = np.mean(off_err)
    print "off_line mean error is:", err

    #通过训练好的模型，判断每种特征对模型的贡献
    print np.sort(zip(etr.feature_importances_, columns),axis=0)

    #预测
    etr.fit(validation_x, validation_y)

    # #评估模型性能
    # etr_test_y = etr.predict(test_x)
    # print 'R-squared value of etr:', etr.score(test_x, test_y)
    # print 'the Mean squared error of etr:', mean_squared_error(ss_y.inverse_transform(test_y), ss_y.inverse_transform(etr_test_y))
    # print 'the mean absolute error of etr:', mean_absolute_error(ss_y.inverse_transform(test_y),ss_y.inverse_transform(etr_test_y))



    #通过训练好的模型进行预测
    etr_predict_y = etr.predict(predict_x)
    etr_predict_y = (ss_y.inverse_transform(etr_predict_y)).astype(int)
    etr_predict_y = pd.DataFrame(etr_predict_y, columns=['day_'+str(i) for i in np.arange(1,8)])

    # 获得预测第二周的样本,并标准化
    etr.fit(train_x, validation_y)
    etr_predict_y2 = etr.predict(predict_x)
    etr_predict_y2 = (ss_y.inverse_transform(etr_predict_y2)).astype(int)
    etr_predict_y2 = pd.DataFrame(etr_predict_y2, columns=['day_' + str(i) for i in np.arange(8, 15)])

    # 将etr_predict_y转为提交格式
    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join([etr_predict_y, etr_predict_y2])
    # result = pd.merge(result, result , on='iid')

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_rf_by_three_weeks.csv', index=False, header=False)

    print result.info()


