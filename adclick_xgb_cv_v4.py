from sklearn.utils import check_array

from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
import datetime as dt

seed = 2017
np.random.seed(seed)

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
df_train = pd.read_csv(base_input_path+'train.csv')
df_test = pd.read_csv(base_input_path+'test.csv')

# Any results you write to the current directory are saved as output.

base_input_path = './data/input/'
base_output_path = './data/output/'

print ("start")
# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')


print ('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print ('The test data has {} rows and {} columns'.format(test.shape[0],test.shape[1]))

# imputing missing values
train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None",inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None",inplace=True)
test['devid'].fillna("None",inplace=True)

# create timebased features

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['tweekday'] = train['datetime'].dt.weekday
test['tweekday'] = test['datetime'].dt.weekday

train['thour'] = train['datetime'].dt.hour
test['thour'] = test['datetime'].dt.hour

train['tminute'] = train['datetime'].dt.minute
test['tminute'] = test['datetime'].dt.minute

# create aggregate features
site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_offer_count_test = test.groupby(['siteid','offerid']).size().reset_index()
site_offer_count_test.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_cat_count_test = test.groupby(['siteid','category']).size().reset_index()
site_cat_count_test.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

site_mcht_count_test = test.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count_test.columns = ['siteid','merchant','site_mcht_count']

# joining all files
agg_df = [site_offer_count, site_cat_count, site_mcht_count]
agg_df_test = [site_offer_count_test, site_cat_count_test, site_mcht_count_test]

for x in agg_df:
    train = train.merge(x)

for x in agg_df_test:
    test = test.merge(x)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# sample 10% data - to avoid memory troubles
# if you have access to large machines, you can use more data for training

train = train.sample(frac=0.1,replace=False) #train.sample(1e6)
print (train.shape)

# select columns to choose
cols_to_use = [x for x in train.columns if x not in list(['ID','datetime','click'])]

# standarise data before training
scaler = StandardScaler().fit(train[cols_to_use])

strain = scaler.transform(train[cols_to_use])
stest = scaler.transform(test[cols_to_use])

# train validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.1, random_state=2017)

print (X_train.shape)
print (X_valid.shape)
print (Y_train.shape)
print (Y_valid.shape)

print("Building model.. ")
# preds, imp, num_boost_rounds,model,auc_score = run_xgboost(X_train, X_valid, Y_train, Y_valid, cols_to_use, 2017)

#xgboost
# prepare dict of params for xgboost to run with
y_mean = np.mean(Y_train)

xgb_params = {
    # 'n_trees': 900,
    # 'n_estimators':1300,
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    # 'eta': 0.005,
    # 'max_depth': 20,
    # 'subsample': 0.95,
    # "tree_method": 'exact',
    'base_score': y_mean, # base prediction = mean(target)
    # 'silent': 1,
    # 'colsample_bytree':0.4,
    "seed": seed,

}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_valid)

dtest_test = xgb.DMatrix(stest)


# xgboost, cross-validation
# cv_result = xgb.cv(xgb_params,
#                    dtrain,
#                    num_boost_round=700, # increase to have better results (~700)
#                    early_stopping_rounds=50,
#                    verbose_eval=50,
#                    show_stdv=False
#                   )
#
# num_boost_rounds = len(cv_result)
# print(num_boost_rounds)

print("GridSearchCV")
xgb_model = xgb.XGBRegressor(xgb_params)

model = GridSearchCV(estimator = xgb_model,param_grid = xgb_params)

model.fit(X_train,Y_train)

print("best xgboost parameters")
print model.best_params_

kfold = KFold(X_train.shape[0],n_folds=5, random_state=2017)
results = cross_val_score(model, dtrain, dtest, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# train model
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# print("Started stacking of models")

'''Train the stacked models then predict the test data'''
#
# stacked_pipeline = make_pipeline(
#     StackingEstimator(estimator=RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)),
#     StackingEstimator(estimator=ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)),
#     StackingEstimator(estimator=LassoLarsCV(normalize=True)),
#     StackingEstimator(
#         estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55,
#                                             min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
#     LassoLarsCV()
#
# )


print("Predicting on actual test data")
y_pred = model.predict(dtest_test, ntree_limit=model.best_iteration + 1)


# stacked_pipeline.fit(X_train[cols_to_use], Y_train)
# results = stacked_pipeline.predict(stest[cols_to_use])

print ("AUC final ")
# check validation accuracy
y__pred_train_xgb =  model.predict(dtest, ntree_limit=model.best_iteration + 1)

roc_xgb_pred = roc_auc_score(y_true = Y_valid, y_score=y__pred_train_xgb)

print("xgb model roc score - {}".format(roc_xgb_pred))

# y_pred_train_stacked= stacked_pipeline.predict(X_train[cols_to_use])
#
# roc_stacked_pred = roc_auc_score(y_true = Y_valid, y_score=y_pred_train_stacked)
#
# print("stacked model roc score - {}".format(roc_stacked_pred))

# y_pred_train_final = y_pred_train_stacked * 0.2855 + y__pred_train_xgb * 0.7145

#
# score= roc_auc_score(y_true = Y_valid, y_score=y_pred_train_final)
#
# print ("score - {}".format(score))

# print ("predicting final click result")
# y_pred_final=average_precision_score(results * 0.2855 + y_pred * 0.7145)

print ("Preparing submit file")

output = pd.DataFrame({'ID': test['ID'], 'click': y_pred})
output.to_csv(base_output_path+'adclick_xgb_cv-v4-auc-{}.csv'.format(roc_xgb_pred), index=False)

print("done")
