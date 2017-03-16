# coding=UTF-8
import pandas as pd
import numpy as np
from features.week_data import get_user_pay_week_data
from features.week_data import get_user_view_week_data
from features.samples_get_by_three_weeks import get_three_weeks_samples
import params

def get_samples():

    # 按周划分最近六周的shop_user_pay样本
    week_data = get_user_pay_week_data()
    week_last_1st = week_data[1]    #倒数第一周
    week_last_2nd = week_data[2]    #倒数第二周
    week_last_3rd = week_data[3]    #倒数第三周
    week_last_4th = week_data[4]    #国庆周剔除
    week_last_5th = week_data[5]    #国庆周剔除
    week_last_6th = week_data[6]
    week_last_7th = week_data[7]
    print week_last_1st, week_last_2nd, week_last_3rd, week_last_6th, week_last_7th

    # 按周划分最近六周的shop_user_pay样本
    view_week_data = get_user_view_week_data()
    view_week_last_1st = view_week_data[1]    #倒数第一周
    view_week_last_2nd = view_week_data[2]    #倒数第二周
    view_week_last_3rd = view_week_data[3]    #倒数第三周
    view_week_last_4th = view_week_data[4]
    view_week_last_5th = view_week_data[5]
    view_week_last_6th = view_week_data[6]
    view_week_last_7th = view_week_data[7]
    #print view_week_last_1st, view_week_last_2nd, view_week_last_3rd, view_week_last_6th, view_week_last_7th

    #计算  浏览/消费  特征
    week_1st_rate = pd.DataFrame(view_week_last_1st.values / (week_last_1st.replace(0, 1)).values.astype(float),
                                 columns=['week_1st_rate_' + str(i) for i in np.arange(7)])
    week_2nd_rate = pd.DataFrame(view_week_last_2nd.values / (week_last_2nd.replace(0, 1)).values.astype(float),
                                 columns=['week_2nd_rate_' + str(i) for i in np.arange(7)])
    week_3rd_rate = pd.DataFrame(view_week_last_3rd.values / (week_last_3rd.replace(0, 1)).values.astype(float),
                                 columns=['week_3rd_rate_' + str(i) for i in np.arange(7)])
    week_6th_rate = pd.DataFrame(view_week_last_6th.values / (week_last_6th.replace(0, 1)).values.astype(float),
                                 columns=['week_6th_rate_' + str(i) for i in np.arange(7)])
    week_7th_rate = pd.DataFrame(view_week_last_7th.values / (week_last_7th.replace(0, 1)).values.astype(float),
                                 columns=['week_7th_rate_' + str(i) for i in np.arange(7)])

    #构造训练样本
    # samples1 = get_three_weeks_samples(week_last_3rd, week_last_4th, week_last_5th)
    # samples2 = get_three_weeks_samples(week_last_2nd, week_last_3rd, week_last_4th)
    # train_x = pd.concat([samples1, samples2])
    # train_y = np.concatenate((np.array(week_last_2nd), np.array(week_last_1st)))
    # train_y = pd.DataFrame(train_y)

    train_x = get_three_weeks_samples(week_last_3rd, week_last_6th, week_last_7th)
    #train_x = train_x.join([view_week_last_3rd, view_week_last_6th, view_week_last_7th])
    train_x = train_x.join([week_3rd_rate, week_6th_rate, week_7th_rate])
    train_y = week_last_2nd

    validation_x = get_three_weeks_samples(week_last_2nd, week_last_3rd, week_last_6th)
    #validation_x = validation_x.join([view_week_last_2nd, view_week_last_3rd, view_week_last_6th])
    validation_x = validation_x.join([week_2nd_rate, week_3rd_rate, week_6th_rate])

    validation_y = week_last_1st

    predict_x = get_three_weeks_samples(week_last_1st, week_last_2nd, week_last_3rd)
    #predict_x = predict_x.join([view_week_last_1st, view_week_last_2nd, view_week_last_3rd])
    predict_x = predict_x.join([week_1st_rate, week_2nd_rate, week_3rd_rate])

    # 将训练样本和测试样本存为csv文件
    train_x.to_csv(params.SAMPLE_PATH  + 'train_x.csv', index=False)
    train_y.to_csv(params.SAMPLE_PATH + 'train_y.csv', index=False)

    validation_x.to_csv(params.SAMPLE_PATH + 'validation_x.csv', index=False)
    validation_y.to_csv(params.SAMPLE_PATH + 'validation_y.csv', index=False)

    predict_x.to_csv(params.SAMPLE_PATH + 'predict_x.csv', index=False)

