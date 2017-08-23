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
from sklearn.cross_validation import train_test_split
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

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)


def run_single(X_train, X_valid, features, target, random_state=0):
    eta = 0.1
    max_depth = 6
    subsample = 1
    colsample_bytree = 1
    min_chil_weight = 1
    start_time = time.time()

    print(
    'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample,
                                                                                          colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight": min_chil_weight,
        "seed": random_state,
        # "num_class" : 22,
    }
    num_boost_round =100
    early_stopping_rounds = 20
    test_size = 0.1

    #X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)

    # area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    check2 = check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)
    score = average_precision_score(X_valid[target].values, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))

    ############################################ ROC Curve



    # # Compute micro-average ROC curve and ROC area
    # fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    # roc_auc = auc(fpr, tpr)
    # # xgb.plot_importance(gbm)
    # # plt.show()
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([-0.02, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # plt.legend(loc="lower right")
    # plt.show()
    # ##################################################


    print('Training time: {} minutes'.format(round((time.time() - start_time) / 60, 2)))
    return test_prediction, imp, gbm.best_iteration + 1 , gbm,score

def run_xgboost(X_train, X_valid,y_train,y_valid, features,random_state=2017):
    eta = 0.1
    max_depth = 6
    subsample = 1
    colsample_bytree = 1
    min_chil_weight = 1
    start_time = time.time()

    print(
        'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample,
                                                                                              colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight": min_chil_weight,
        "seed": random_state,
        # "num_class" : 22,
    }
    num_boost_round = 100
    early_stopping_rounds = 20
    test_size = 0.1

    # X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    # print('Length train:', len(X_train.index))
    # print('Length valid:', len(X_valid.index))

    # y_train = X_train[target]
    # y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)

    # area under the precision-recall curve
    score = average_precision_score(y_valid, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    check2 = check.round()
    score = precision_score(y_valid, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(y_valid, check2)
    print('recall score: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)
    score = average_precision_score(y_valid, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))


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

#new featrures
site_ad_count = train.groupby(['siteid','ID']).size().reset_index()
site_ad_count.columns = ['siteid','ID','site_ad_count']

site_ad_count_test = test.groupby(['siteid','ID']).size().reset_index()
site_ad_count_test.columns = ['siteid','ID','site_ad_count']

site_device_count = train.groupby(['siteid','devid']).size().reset_index()
site_device_count.columns = ['siteid','devid','site_device_count']

site_device_count_test = test.groupby(['siteid','devid']).size().reset_index()
site_device_count_test.columns = ['siteid','devid','site_device_count']

device_id_count = train.groupby(['devid','ID']).size().reset_index()
device_id_count.columns = ['devid','ID','device_id_count']

device_id_count_test = test.groupby(['devid','ID']).size().reset_index()
device_id_count_test.columns = ['devid','ID','device_id_count']


# joining all files
agg_df = [site_offer_count, site_cat_count, site_mcht_count,site_ad_count,site_device_count,device_id_count]
agg_df_test = [site_offer_count_test, site_cat_count_test, site_mcht_count_test,site_ad_count_test,site_device_count_test,device_id_count_test]

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
    'n_trees': 500,
    'n_estimators':1300,
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    'eta': 0.005,
    'max_depth': 20,
    'subsample': 0.95,
    "tree_method": 'exact',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'colsample_bytree':0.4,
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

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=1200)

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
output.to_csv(base_output_path+'adclick_xgb-v4-auc-{}.csv'.format(roc_xgb_pred), index=False)

print("done")
