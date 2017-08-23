#script from https://github.com/HackerEarth-Challenges/ML-Challenge-3/blob/master/CatBoost_Starter.ipynb
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

base_input_path = './data/input/'
base_output_path = './data/output/'

print ("start")
# read datasets
train = pd.read_csv(base_input_path+'train.csv')
test = pd.read_csv(base_input_path+'test.csv')

# check missing values per column
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



# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')

cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

# catboost accepts categorical variables as indexes
cat_cols = [0,4,5,8,9]#[0,1,2,4,6,7,8]

# modeling on sampled (1e6) rows
# rows = np.random.choice(train.index.values, 1e6)
# sampled_train = train.loc[rows]

# rows = np.random.choice(train.index.values, 1e6)
sampled_train = train.sample(frac=0.6,replace=False)

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

# trainX = train[cols_to_use]
# trainY = train['click']

print (trainX.head())

print ("catboost training")

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.5)
model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1, eval_metric='AUC', random_seed=1)

model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

# check validation accuracy
vpreds = model.predict_proba(X_train)[:,1]
score=roc_auc_score(y_true = y_train, y_score=vpreds)

print ("score - {}".format(score))


print ("predicting..")
pred = model.predict_proba(test[cols_to_use])[:,1]

print ("writing submit file")

sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv(base_output_path+'adclick_catboost_v1_{}.csv'.format(score),index=False)

print ("done")