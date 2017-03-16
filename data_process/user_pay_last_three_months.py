import cPickle
import params
from data_process.get_last_three_months import get_last_three_months

def user_pay_last_three_months():
    data_path = params.DATA_PATH + 'user_pay.txt'
    data = get_last_three_months(data_path)
    #print data

    f = open(params.DATA_PATH + "user_pay_last_three_months.pkl", 'wb')
    cPickle.dump(data.astype(int), f, -1)
    f.close()