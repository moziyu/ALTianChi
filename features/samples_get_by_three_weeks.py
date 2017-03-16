# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import params

def get_three_weeks_samples(first_week, second_week, third_week):
    #print week_last_1st, week_last_2nd, week_last_3rd, week_last_6th
    data_x = third_week.join([second_week, first_week])
    data_x.columns = ['data_' + str(i) for i in np.arange(data_x.shape[1])]

    #将商家三周的各统计特征作为训练样本的特征
    data_sum = data_x.sum(axis=1)
    data_mean = data_x.mean(axis=1)
    data_weekend = ['data_5','data_6','data_12','data_13','data_19','data_20']
    data_ratio_wk = (data_x[data_weekend]).sum(axis=1)/(data_sum.replace(0,1))
    data_std = data_x.std(axis=1)
    data_max = data_x.max(axis=1)
    data_min = data_x.min(axis=1)
    data_median = data_x.median(axis=1)
    data_mad = data_x.mad(axis=1)
    data_var = data_x.var(axis=1)

    #对后两周流量进行跨度为三天的中心移动平均
    move_mean = data_x.iloc[:,7:21].rolling(window=3,axis=1).mean()
    move_mean = move_mean.dropna(axis=1)
    move_mean.columns = ['move_mean_' + str(j) for j in np.arange(12)]
    #对后两周流量跨度为一周的中心移动平均
    moveweek_mean = data_x.iloc[:, 7:21].rolling(window=7, axis=1).mean()
    moveweek_mean = moveweek_mean.dropna(axis=1)
    moveweek_mean.columns = ['moveweek_mean_' + str(j) for j in np.arange(8)]

    # 流量的差分特征
    week_diff = second_week.values - third_week.values
    week_diff = pd.DataFrame(week_diff, columns=['week_diff_' + str(i) for i in np.arange(7)])
    day_diff = data_x.values[:, 7:20] - data_x.values[:, 8:21]
    day_diff = pd.DataFrame(day_diff, columns=['day_diff_' + str(i) for i in np.arange(13)])
    two_days_diff = data_x.values[:, 7:19] - data_x.values[:, 9:21]
    two_days_diff = pd.DataFrame(two_days_diff, columns=['two_days_diff_' + str(i) for i in np.arange(12)])

    #对后一周流量进行对数变换
    data_log = np.log1p(data_x.values[:,14:21])
    data_log = pd.DataFrame(data_log, columns=['log_'+str(i) for i in range(7)])

    #商家的属性特征
    shop_features = pd.read_pickle(params.SAMPLE_PATH + "shop_features.pkl")

    ####################################################################################################################
    #生成多项式特征
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    poly_x = poly.fit_transform(data_x.iloc[:, 7:21])
    poly_x = pd.DataFrame(poly_x, columns=['poly_'+str(i) for i in range(poly_x.shape[1])])   #将train_x变成DataFrame类型
    #data_x = data_x.iloc[:,0:7].join(poly_x)

    ####################################################################################################################
    #将构造的统计特征特征加入样本
    data_x['sum'] = data_sum
    data_x['ratio_wk'] = data_ratio_wk
    data_x['mean'] = data_mean
    data_x['std'] = data_std
    data_x['max'] = data_max
    data_x['min'] = data_min
    data_x['median'] = data_median
    data_x['mad'] = data_mad
    data_x['var'] = data_var

    #将移动平均特征、差分特征以及lg对换特征加入样本
    #data_x = data_x.join(move_mean)
    #data_x = data_x.join( moveweek_mean)
    data_x = data_x.join([week_diff, day_diff])
    #data_x = data_x.join(data_log)
    data_x = data_x.join( two_days_diff)

    #以shop_info构造商家特征,加入训练样本
    data_x = shop_features.join(data_x)
    data_x['revenue'] = data_x['mean'] * data_x['per_pay']   #平均销售额

    #去掉商家的iid
    data_x = data_x.drop('iid', axis=1)

    return data_x



