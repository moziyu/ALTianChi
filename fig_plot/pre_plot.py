# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import params
#要查看的店铺编号

def pre_plot():

    data_pay = pd.read_pickle(params.DATA_PATH + 'user_pay_last_three_months.pkl')
    data_view = pd.read_pickle(params.DATA_PATH + 'user_view_last_three_months.pkl')

    #要查看的时期(可查看2016/08/01-2.16/10/31)
    colunms = range(1,93)

    #要查看的商家
    index =np.random.permutation(range(1,2001)) #随机查看

    for i in index:
        data1 = data_pay.iloc[i,colunms]
        data2 = data_view.iloc[i, colunms]
        #plt.figure(i+1)
        plt.subplot(2,1,1)
        data1.plot()
        plt.subplot(2,1,2)
        data2.plot()
        print data1
        plt.show()