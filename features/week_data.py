# coding=UTF-8
import pandas as pd
import params

def get_week_data(data):

    # 以周为单位构造样本
    shape = data.shape
    weeks_num = (shape[1]-1) / 7
    week_data = []
    for index in range(weeks_num):
        x = data.iloc[:, shape[1] - index * 7: shape[1] - (index - 1) * 7]
        week_data.append(x)

    return week_data

def get_user_pay_week_data():
    data = pd.read_pickle(params.DATA_PATH + "user_pay_last_three_months.pkl")
    return get_week_data(data)

def get_user_view_week_data():
    data = pd.read_pickle(params.DATA_PATH + "user_view_last_three_months.pkl")
    return get_week_data(data)
