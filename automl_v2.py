from sklearn.preprocessing import LabelEncoder, StandardScaler

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
import lightgbm as lgb
from tpot import TPOTClassifier

base_input_path = './data/input/'
base_output_path = './data/output/'

# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')


# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ", start_time)


# check missing values per column
print("Missing record information")
#check missing values per column
train.isnull().sum(axis=0)/train.shape[0]

# impute missing values

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)

# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


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
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.5, random_state=2017)

print (X_train.shape)
print (X_valid.shape)
print (Y_train.shape)
print (Y_valid.shape)


# cols = ['siteid','offerid','category','merchant']
#
# for x in cols:
#     df_train[x] = df_train[x].astype('object')
#     df_test[x] = df_test[x].astype('object')
#
# cat_cols = cols + ['countrycode','browserid','devid']
#
#
# # process columns, apply LabelEncoder to categorical features
# for c in cat_cols:
#     lbl = LabelEncoder()
#     lbl.fit(list(df_train[c].values) + list(df_test[c].values))
#     df_train[c] = lbl.transform(list(df_train[c].values))
#     df_test[c] = lbl.transform(list(df_test[c].values))
#
# # shape
# print('Shape train: {}\nShape test: {}'.format(df_train.shape, df_test.shape))
#
# features = list(df_train.columns.values) #list(set(train.columns) - set(['ID','datetime','click']))
# features.remove('ID')
# features.remove('datetime')
# features.remove('click')
# print(features)


# TPOT analysis
# df_train_new = df_train[features];

click_train = train['click'].values
# print ("train test split")
#
# training_indices, validation_indices = training_indices, testing_indices = train_test_split(df_train[features].index, stratify = click_train, train_size=0.75, test_size=0.25)
# training_indices.size, validation_indices.size
#
# print (training_indices)

print ("fitting TPOT classifier")
tpot = TPOTClassifier(generations=100,verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=100)
tpot.fit(X_train, Y_train)

print ("scoring TPOT classifier")
tpot_score = tpot.score(X_valid, Y_valid)

print ("exporting TPOT classifier")
tpot.export('tpot__h_ml3_pipeline.py')

print("Predicting on actual test data")
# y_pred = model.predict(xgb.DMatrix(df_test[features]), ntree_limit=model.best_iteration + 1)
y_pred = tpot.predict(stest)


# y_pred, imp, num_boost_rounds,auc_score = run_single(df_train, df_test, features, 'click', 42)

print ("Preparing submit file")

output = pd.DataFrame({'ID': test['ID'], 'click': y_pred})
output.to_csv(base_output_path+'tpot_v1_score_{tpot_score}.csv'.format(tpot_score), index=False)

print ("done")

