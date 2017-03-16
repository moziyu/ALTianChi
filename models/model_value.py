# coding=UTF-8
import pandas as pd

def value_mode(pre, real):
    if (len(pre.shape) == 1):
        pre = pd.DataFrame(pre, columns=[0])
        real = pd.DataFrame(real, columns=[0])
    else:
        pre = pd.DataFrame(pre, columns=[i for i in range(pre.shape[1])])
        real = pd.DataFrame(real, columns=[i for i in range(real.shape[1])])

    if (len(pre) != len(real)):
        print 'len(pre)!=len(real)', '\n'
    if (len(pre.columns) != len(real.columns)):
        print 'len(pre.columns)!=len(real.columns)', '\n'
    N = (pre.shape)[0]  # N：商家总数
    T = (pre.shape)[1]
    #print 'N:', N, '\t', 'T:', T, '\n'

    L = 0
    for t in range(T):
        for n in range(N):
            c_it = round(pre.ix[n, t])  # c_it：预测的客流量
            c_git = round(real.ix[n, t])  # c_git：实际的客流量
            if ((c_it == 0 and c_git == 0) or (c_it + c_git) == 0):
                c_it = 1
                c_git = 1
            L = L + abs((float(c_it) - c_git) / (c_it + c_git))
    #print "off_line error is:", L / (N * T)
    return L / (N * T)