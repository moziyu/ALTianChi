from models.xgboost_model_by_three_weeks import xg_regress_by_three_weeks
from features.get_samples import get_samples

get_samples()
xg_regress_by_three_weeks()