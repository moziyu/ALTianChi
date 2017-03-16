# coding=UTF-8
import pandas as pd
import numpy as np

def get_date(time):
    return time.split(' ')[0]

def get_last_three_months(data_path):
    # 获取倒数三个月的数据
    data = pd.read_csv(data_path, header=None)
    data.columns = ['uid', 'iid', 'time']
    data = data[data['time'].astype(str) >= '2016-08']

    # 横轴日期，纵轴商家ID统计流量
    data = data[data['time'] >= '2016-08']
    data['time'] = data['time'].apply(get_date)
    data = data.groupby(['iid','time'],as_index=False).count()
    result = pd.DataFrame(np.arange(1,2001),columns=['iid'])

    for index in range(1,32):
        if index<10 :
            date = '2016-08-0' + str(index)
        else:
            date = '2016-08-' + str(index)
        result[date] = np.zeros((result.shape[0],1))
    for index in range(1, 31):
        if index<10 :
            date = '2016-09-0' + str(index)
        else:
            date = '2016-09-' + str(index)
        result[date] = np.zeros((result.shape[0],1))
    for index in range(1, 32):
        if index<10 :
            date = '2016-10-0' + str(index)
        else:
            date = '2016-10-' + str(index)
        result[date] = np.zeros((result.shape[0],1))
    for row in data.values:
        result.loc[row[0] - 1,row[1]] = row[2]

    # 对异常值进行处理
    shape = result.shape
    for i in range(shape[0]):
        for index in range(shape[1]):
            median = result.iloc[i, :].median()
            mean = result.iloc[i, :].mean()
            place = max(median, mean)
            if (result.iloc[i, index] <= 1 or result.iloc[i, index] > 5*mean):
                result.iloc[i, index] = place
            if (result.iloc[i, index] <= median/5.0 or result.iloc[i, index] > 5*mean):
                result.iloc[i, index] = place

    return result
