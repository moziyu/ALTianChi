# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import params

def result_plot():

    data = pd.read_pickle(params.DATA_PATH + 'user_pay_last_three_months.pkl')
    result1 = pd.read_csv(params.OUTPUT_PATH + 'result_etr_by_three_weeks.csv',names=range(0,15))
    result2 = pd.read_csv(params.OUTPUT_PATH + 'result_etr_gs_by_three_weeks.csv', names=range(0, 15))

    #要查看的时期(可查看2016/08/01-2.16/10/31)
    data_colunms = range(86, 93)
    result1_columns = range(1, 8)
    result2_columns = range(1, 8)

    #要查看的商家
    shop_iid = np.arange(1,100)
    for iid in shop_iid:
        data_plot = data.iloc[iid-1,data_colunms]
        result1_plot = result1.iloc[iid-1, result1_columns]
        result2_plot = result2.iloc[iid-1, result2_columns]

        #data_plot.plot()
        plt.plot(range(1,8),data_plot)
        result1_plot.plot()
        result2_plot.plot()

        print data_plot
        print result1_plot
        print result2_plot

        plt.show()