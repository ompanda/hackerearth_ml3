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

# train validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(strain, train.click, test_size = 0.1, random_state=2017)

print (X_train.shape)
print (X_valid.shape)
print (Y_train.shape)
print (Y_valid.shape)


# model architechture
def keras_model(train):
    input_dims = train.shape[1]
    classes = 2

    model = Sequential()
    # model.add(Dense(100, activation='relu', input_shape=(input_dims,)))  # layer 1
    # model.add(Dense(30, activation='relu'))  # layer 2
    # model.add(Dense(classes, activation='sigmoid'))  # output
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model = Sequential()
    # Input layer with dimension input_dims and hidden layer i with input_dims neurons.
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims // 2, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    #
    # Output Layer.
    model.add(Dense(classes, activation='sigmoid'))  # output

    # # Use a large learning rate with decay and a large momentum.
    # # Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # # compile this model
    model.compile(loss='binary_crossentropy',  # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=["accuracy"]  # you can add several if needed
                  )

    # Visualize NN architecture
    print(model.summary())

    return model


callback = EarlyStopping(monitor='val_acc', patience=3)

# one hot target columns
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)

print("init model")
# train model
model = keras_model(X_train)

no_of_epochs = 50
scores_for_epochs= dict()

print("fitting model")
# for no_of_epoch in range(1,no_of_epochs,50):
#     print ("epoch - {}".format(no_of_epoch))
model.fit(X_train, Y_train, 1000, no_of_epochs, callbacks=[callback],validation_data=(X_valid, Y_valid),shuffle=True)

# check validation accuracy
print("cross valdiation summary")
vpreds = model.predict_proba(X_valid)[:,1]
score=roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)

# print ("epoch - {} - score - {}".format(no_of_epoch,score))
# scores_for_epochs[no_of_epoch]= score

print("auc score is - {}".format(score))

# predict on test data
print("predicting on test data")
test_preds = model.predict_proba(stest)[:,1]

#create submission file
print("preparing submit file")
submit = pd.DataFrame({'ID':test.ID, 'click':test_preds})
submit.to_csv(base_output_path+'adclick_keras_v2_{}.csv'.format(score), index=False)

print ("done")

import matplotlib.pyplot as plt

# fig_loss = plt.figure(figsize=(100, 100))
# plt.plot(scores_for_epochs.keys())
# plt.plot(scores_for_epochs.values())
# plt.title('model loss')
# plt.ylabel('auc score')
# plt.xlabel('epoch')
# plt.legend(['epoch', 'score'], loc='upper left')
# plt.show()
# fig_loss.savefig("model_loss.png")