from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.constraints import maxnorm
from math import exp, log, sqrt
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

seed = 2017
np.random.seed(seed)

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

#train = train.sample(frac=0.8,replace=False) #train.sample(1e6)
print (train.shape)

# select columns to choose
cols_to_use = [x for x in train.columns if x not in list(['ID','datetime','click'])]

# standarise data before training
scaler = StandardScaler().fit(train[cols_to_use])

strain = scaler.transform(train[cols_to_use])
stest = scaler.transform(test[cols_to_use])

y= train.click

# # train validation split
# X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.5, random_state=2017)
#
# print (X_train.shape)
# print (X_valid.shape)
# print (Y_train.shape)
# print (Y_valid.shape)
#

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# model architechture
def keras_model(input_dims):
   # input_dims = train.shape[1]
    classes = 2

    # model = Sequential()
    # model.add(Dense(100, activation='relu', input_shape=(input_dim,)))  # layer 1
    # model.add(Dense(30, activation='relu'))  # layer 2
    # model.add(Dense(classes, activation='sigmoid'))  # output
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model = Sequential()
    # Input layer with dimension input_dims and hidden layer i with input_dims neurons.
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims // 2, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))

    # Output Layer.
    model.add(Dense(classes, activation='sigmoid'))  # output

    # Use a large learning rate with decay and a large momentum.
    # Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile this model
    model.compile(loss='binary_crossentropy',  # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=["accuracy", "mse"]  # you can add several if needed
                  )

    # Visualize NN architecture
    print(model.summary())

    return model


callback = EarlyStopping(monitor='val_acc', patience=3)

# # one hot target columns
# Y_train = to_categorical(Y_train)
# Y_valid = to_categorical(Y_valid)

# # train model
# model = keras_model(X_train)
# model.fit(X_train, Y_train, 1000, 50, callbacks=[callback],validation_data=(X_valid, Y_valid),shuffle=True)


# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=keras_model(strain.shape[1]),
    nb_epoch=300,
    batch_size=30,
    verbose=1
)


n_splits = 5
print ("k - {} - fold validation".format(n_splits))
kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
kf.get_n_splits(strain)

roc_auc_scores = list()

for fold, (train_index, test_index) in enumerate(kf.split(strain)):

    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_val = strain[train_index], strain[test_index]
    y_tr, y_val = y[train_index], y[test_index]

    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            patience=3,
            verbose=1)
    ]
    # fit estimator
    history = estimator.fit(
        X_tr,
        y_tr,
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks,
        shuffle=True
    )

    pred = estimator.predict(X_val)

    score = roc_auc_score(y_val, estimator.predict(X_val))
    roc_auc_scores.append(score)

    print('Fold %d: roc auc %f'%(fold, score))

mean_roc_auc = mean(roc_auc_scores)



print('=====================')
print( 'Mean roc auc score %f'%mean_roc_auc)
print('=====================')


# # check validation accuracy
# vpreds = estimator.predict_proba(X_valid)[:,1]
# score=roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)

print ("score - {}".format(mean_roc_auc))

# predict on test data
test_preds = estimator.predict_proba(stest)[:,1]

#create submission file
submit = pd.DataFrame({'ID':test.ID, 'click':test_preds})
submit.to_csv(base_output_path+'adclick_keras_v3_{}.csv'.format(score), index=False)

print ("done")
