from sklearn.preprocessing import LabelEncoder

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
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


base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
df_train = pd.read_csv(base_input_path+'train.csv')
df_test = pd.read_csv(base_input_path+'test.csv')


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


def run_single(train, test, features, target, random_state=0):
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

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=np.NAN)
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing=np.NAN)

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
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing=np.NAN), ntree_limit=gbm.best_iteration + 1)
    score = average_precision_score(test[target].values, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))

    ############################################ ROC Curve



    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    # xgb.plot_importance(gbm)
    # plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    ##################################################


    print('Training time: {} minutes'.format(round((time.time() - start_time) / 60, 2)))
    return test_prediction, imp, gbm.best_iteration + 1 , gbm,score


# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ", start_time)


# process columns, apply LabelEncoder to categorical features
for c in df_train.columns:
    if df_train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df_train[c].values) + list(df_test[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))
        df_test[c] = lbl.transform(list(df_test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(df_train.shape, df_test.shape))


train, test = train_test_split(df_train, test_size=.1, random_state=random.seed(2016))

features = list(train.columns.values)
features.remove('click')
print(features)

print("Building model.. ", dt.datetime.now() - start_time)
preds, imp, num_boost_rounds,model,auc_score = run_single(train, test, features, 'click', 42)

print("Predicting on actual test data")
y_pred = model.predict(xgb.DMatrix(df_test[features],missing=np.NAN), ntree_limit=model.best_iteration + 1)


# y_pred, imp, num_boost_rounds,auc_score = run_single(df_train, df_test, features, 'click', 42)

print ("Preparing submit file")

output = pd.DataFrame({'ID': df_test['ID'], 'click': y_pred})
output.to_csv(base_output_path+'xgboost-v1-auc-{}.csv'.format(auc_score), index=False)

print(dt.datetime.now() - start_time)

#
# # process columns, apply LabelEncoder to categorical features
# for c in train.columns:
#     if train[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(train[c].values) + list(test[c].values))
#         train[c] = lbl.transform(list(train[c].values))
#         test[c] = lbl.transform(list(test[c].values))
#
# # shape
# print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
#
#
#
#
#
#
#
# y_train = train["y"]
# y_mean = np.mean(y_train)
#
#
# #Preparing Regressor
#
# # prepare dict of params for xgboost to run with
# xgb_params = {
#     'n_trees': 500,
#     'eta': 0.005,
#     'max_depth': 4,
#     'subsample': 0.95,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'base_score': y_mean, # base prediction = mean(target)
#     'silent': 1
# }
#
# # form DMatrices for Xgboost training
# dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
# dtest = xgb.DMatrix(test)
#
# # xgboost, cross-validation
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
#
# # train model
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#
# # check f2-score (to get higher score - increase num_boost_round in previous cell)
#
#
# r2_score =r2_score(dtrain.get_label(), model.predict(dtrain))
#
# print(r2_score)
#
# # make predictions and save results
# y_pred = model.predict(dtest)
# output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
# output.to_csv(base_output_path+'xgboost-v3-r2-{0}-pca-ica.csv'.format(r2_score), index=False)