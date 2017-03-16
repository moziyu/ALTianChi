# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import params
import model_value


def lr_regress_by_three_weeks():
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
    # 训练模型
    lr_test_y = []
    lr_predict_y = []

    for i in range(0, 7):
        #lr = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0])
        lr =linear_model.LassoCV(alphas=[0.01])
        lr.fit(train_x, train_y[:, i])
        lr_test_y.append(lr.predict(validation_x))
        lr_predict_y.append(lr.predict(predict_x))
    lr_test_y = np.transpose(lr_test_y)
    lr_predict_y = np.transpose(lr_predict_y)

    #评估模型性能
    # print 'R-squared value of gbr:', gbr.score(test_x, test_y)
    # print 'the Mean squared error of gbr:', mean_squared_error(ss_y.inverse_transform(test_y), ss_y.inverse_transform(gbr_test_y))
    # print 'the mean absolute error of gbr:', mean_absolute_error(ss_y.inverse_transform(test_y),ss_y.inverse_transform(gbr_test_y))
    print lr_test_y.shape
    print validation_y.shape
    print "off_line error is:", model_value.value_mode(ss_y.inverse_transform(lr_test_y), ss_y.inverse_transform(validation_y))   #线下误差

    # 通过训练好的模型，判断每种特征对模型的贡献
    #print np.sort(zip(gbr.feature_importances_, columns), axis=0)

    # # 将etr_predict_y转为提交格式
    lr_predict_y = (ss_y.inverse_transform(lr_predict_y)).astype(int)
    lr_predict_y = pd.DataFrame(lr_predict_y)

    result = pd.DataFrame(np.arange(1, 2001), columns=['iid'])
    result = result.join(lr_predict_y)
    result = pd.merge(result, result, on='iid')

    if (not os.path.exists(params.OUTPUT_PATH)):
        os.mkdir(params.OUTPUT_PATH)
    result.to_csv(params.OUTPUT_PATH + 'result_lr_by_three_weeks.csv', index=False, header=False)

    print result


