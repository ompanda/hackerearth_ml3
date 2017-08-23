from sklearn.preprocessing import LabelEncoder

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
df_train = pd.read_csv(base_input_path+'train.csv')
df_test = pd.read_csv(base_input_path+'test.csv')


# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ", start_time)


# check missing values per column
print("Missing record information")
print(df_train.isnull().sum(axis=0)/df_train.shape[0])


df_train['siteid'].fillna(-999, inplace=True)
df_test['siteid'].fillna(-999, inplace=True)

df_train['browserid'].fillna("None", inplace=True)
df_test['browserid'].fillna("None", inplace=True)

df_train['devid'].fillna("None", inplace=True)
df_test['devid'].fillna("None", inplace=True)


# set datatime
df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])


# create datetime variable
df_train['tweekday'] = df_train['datetime'].dt.weekday
df_train['thour'] = df_train['datetime'].dt.hour
df_train['tminute'] = df_train['datetime'].dt.minute

df_test['tweekday'] = df_test['datetime'].dt.weekday
df_test['thour'] = df_test['datetime'].dt.hour
df_test['tminute'] = df_test['datetime'].dt.minute


cols = ['siteid','offerid','category','merchant']

for x in cols:
    df_train[x] = df_train[x].astype('object')
    df_test[x] = df_test[x].astype('object')

cat_cols = cols + ['countrycode','browserid','devid']


# process columns, apply LabelEncoder to categorical features
for c in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(df_train[c].values) + list(df_test[c].values))
    df_train[c] = lbl.transform(list(df_train[c].values))
    df_test[c] = lbl.transform(list(df_test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(df_train.shape, df_test.shape))

features = list(df_train.columns.values) #list(set(train.columns) - set(['ID','datetime','click']))
features.remove('ID')
features.remove('datetime')
features.remove('click')
print(features)


# TPOT analysis
# df_train_new = df_train[features];

click_train = df_train['click'].values
# print ("train test split")
#
# training_indices, validation_indices = training_indices, testing_indices = train_test_split(df_train[features].index, stratify = click_train, train_size=0.75, test_size=0.25)
# training_indices.size, validation_indices.size
#
# print (training_indices)

print ("fitting TPOT classifier")
tpot = TPOTClassifier(generations=1,verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=1)
tpot.fit(df_train[features], click_train)

print ("scoring TPOT classifier")
tpot_score = tpot.score(df_train[features], click_train)

print ("exporting TPOT classifier")
tpot.export('tpot__h_ml3_pipeline.py')

print("Predicting on actual test data")
# y_pred = model.predict(xgb.DMatrix(df_test[features]), ntree_limit=model.best_iteration + 1)
y_pred = tpot.predict(df_test[features])


# y_pred, imp, num_boost_rounds,auc_score = run_single(df_train, df_test, features, 'click', 42)

print ("Preparing submit file")

output = pd.DataFrame({'ID': df_test['ID'], 'click': y_pred})
output.to_csv(base_output_path+'tpot_v1_score_{tpot_score}.csv'.format(tpot_score), index=False)

print ("done")

