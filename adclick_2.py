from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
from sklearn.base import BaseEstimator
from keras.layers import Input, Embedding, Dense, Flatten, merge, Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
# from keras import initializations
import itertools
import pandas as pd



def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def batch_generator(X, y, batch_size=128, shuffle=True):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = [X[i][batch_ids] for i in range(len(X))]
            y_batch = y[batch_ids]
            yield X_batch, y_batch

def predict_batch(model, X_t, batch_size=128):
    outcome = []
    for X_batch, y_batch in test_batch_generator(X_t, np.zeros(X_t[0].shape[0]), batch_size=batch_size):
        outcome.append(model.predict(X_batch, batch_size=batch_size))
    outcome = np.concatenate(outcome).ravel()
    return outcome


def build_model(max_features, K=8, solver='adam', l2=0.0, l2_fm=0.0):
    inputs = []
    flatten_layers = []
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32', name='input_%s' % c)
        num_c = max_features[c]

        embed_c = Embedding(
            num_c,
           str(K),
            input_length=1,
            name='embed_%s' % c,
            W_regularizer=l2_reg(l2_fm)
        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []
    for emb1, emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = merge([emb1, emb2], mode='dot', dot_axes=1)
        fm_layers.append(dot_layer)

    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
            num_c,
            1,
            input_length=1,
            name='linear_%s' % c,
            W_regularizer=l2_reg(l2)
        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)

    print ("fm layers")
    print (fm_layers)
    flatten = merge(fm_layers, mode='sum')
    outputs = Activation('sigmoid', name='outputs')(flatten)

    model = Model(input=inputs, output=outputs)

    model.compile(
        optimizer=solver,
        loss='binary_crossentropy'
    )

    return model


class KerasFM(BaseEstimator):
    def __init__(self, max_features=[], K=8, solver='adam', l2=0.0, l2_fm=0.0):
        self.model = build_model(max_features, K, solver, l2=l2, l2_fm=l2_fm)

    def fit(self, X, y, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_data=None):
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=shuffle, verbose=verbose,
                       validation_data=None)

    def fit_generator(self, X, y, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_data=None,
                      callbacks=None):
        tr_gen = batch_generator(X, y, batch_size=batch_size, shuffle=shuffle)
        if validation_data:
            X_test, y_test = validation_data
            te_gen = batch_generator(X_test, y_test, batch_size=batch_size, shuffle=False)
            nb_val_samples = X_test[-1].shape[0]
        else:
            te_gen = None
            nb_val_samples = None

        self.model.fit_generator(
            tr_gen,
            samples_per_epoch=X[-1].shape[0],
            nb_epoch=nb_epoch,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=te_gen,
            nb_val_samples=nb_val_samples,
            max_q_size=10
        )

    def predict(self, X, batch_size=128):
        y_preds = predict_batch(self.model, X, batch_size=batch_size)
        return y_preds


print("start")

base_input_path = './data/input/'
base_output_path = './data/output/'

train = pd.read_csv(base_input_path+"train.csv")
test = pd.read_csv(base_input_path+"test.csv")

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# sample 10% data - to avoid memory troubles
# if you have access to large machines, you can use more data for training

train = train.sample(frac=0.8,replace=False) #train.sample(1e6)
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


y_train = train['click']
train = train.drop('click',axis=1)

print("starting keras FM")
keras_fm = KerasFM(max_features=cols_to_use)

print("fit generator")
keras_fm.fit_generator(train,y_train)

# check validation accuracy
y_train_preds = keras_fm.predict_proba(stest)[:,1]
score=roc_auc_score(y_true = y_train[:,1], y_score=y_train_preds)
print("roc auc score - {}".format(score))

print("predict")
y_test = keras_fm.predict(test)

#create submission file
submit = pd.DataFrame({'ID':test.ID, 'click':y_test})
submit.to_csv(base_output_path+'adclick_keras_fm_v2_{}.csv'.format(score), index=False)

print ("done")

