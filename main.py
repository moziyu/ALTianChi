from data_process.user_pay_last_three_months import user_pay_last_three_months
from data_process.user_view_last_three_months import user_view_last_three_months
from features.shop_features import get_shop_feature
from features.get_samples import get_samples
from features.shop_clusters import shop_clusters
from models.ert_regress import ert_regress

print "######step1: running user_pay_last_three_months.py"
user_pay_last_three_months()

print "######step2: running user_view_last_three_months.py"
user_view_last_three_months()

print "######setp3: running shop_features.py"
get_shop_feature()

print "######step4: running get_samples.py"
get_samples()

print "######step5: running shop_clusters.py"
shop_clusters()

print "######step6: running ert_regress.py"
ert_regress()




