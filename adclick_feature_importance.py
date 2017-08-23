import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

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

print("adding new features")
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

#second batch feature add
device_offer_count = train.groupby(['devid','offerid']).size().reset_index()
device_offer_count.columns = ['devid','offerid','device_offer_count']

device_offer_count_test = test.groupby(['devid','offerid']).size().reset_index()
device_offer_count_test.columns = ['devid','offerid','device_offer_count']

merchant_device_count = train.groupby(['merchant','devid']).size().reset_index()
merchant_device_count.columns = ['merchant','devid','merchant_device_count']

merchant_device_count_test = test.groupby(['merchant','devid']).size().reset_index()
merchant_device_count_test.columns = ['merchant','devid','merchant_device_count']

category_ad_count = train.groupby(['category','ID']).size().reset_index()
category_ad_count.columns = ['category','ID','category_ad_count']

category_ad_count_test = test.groupby(['category','ID']).size().reset_index()
category_ad_count_test.columns = ['category','ID','category_ad_count']

browser_ad_count = train.groupby(['browserid','ID']).size().reset_index()
browser_ad_count.columns = ['browserid','ID','browser_ad_count']

browser_ad_count_test = test.groupby(['browserid','ID']).size().reset_index()
browser_ad_count_test.columns = ['browserid','ID','browser_ad_count']

country_ad_count = train.groupby(['countrycode','ID']).size().reset_index()
country_ad_count.columns = ['countrycode','ID','country_ad_count']

country_ad_count_test = test.groupby(['countrycode','ID']).size().reset_index()
country_ad_count_test.columns = ['countrycode','ID','country_ad_count']

country_offer_count = train.groupby(['countrycode','offerid']).size().reset_index()
country_offer_count.columns = ['countrycode','offerid','country_offer_count']

country_offer_count_test = test.groupby(['countrycode','offerid']).size().reset_index()
country_offer_count_test.columns = ['countrycode','offerid','country_offer_count']

country_site_count = train.groupby(['countrycode','siteid']).size().reset_index()
country_site_count.columns = ['countrycode','siteid','country_site_count']

country_site_count_test = test.groupby(['countrycode','siteid']).size().reset_index()
country_site_count_test.columns = ['countrycode','siteid','country_site_count']

country_browser_count = train.groupby(['countrycode','browserid']).size().reset_index()
country_browser_count.columns = ['countrycode','browserid','country_browser_count']

country_browser_count_test = test.groupby(['countrycode','browserid']).size().reset_index()
country_browser_count_test.columns = ['countrycode','browserid','country_browser_count']

#added new feature
device_browser_count = train.groupby(['devid','browserid']).size().reset_index()
device_browser_count.columns = ['devid','browserid','device_browser_count']

device_browser_count_test = test.groupby(['devid','browserid']).size().reset_index()
device_browser_count_test.columns = ['devid','browserid','device_browser_count']

# # joining all files
# agg_df = [site_offer_count, site_cat_count, site_mcht_count,site_ad_count,site_device_count,device_id_count]
# agg_df_test = [site_offer_count_test, site_cat_count_test, site_mcht_count_test,site_ad_count_test,site_device_count_test,device_id_count_test]

# # joining all files
agg_df = [site_offer_count, site_cat_count, site_mcht_count,site_ad_count,site_device_count,device_id_count,device_offer_count,merchant_device_count,
          category_ad_count,browser_ad_count,country_ad_count,country_offer_count,country_site_count,country_browser_count,device_browser_count]
agg_df_test = [site_offer_count_test, site_cat_count_test, site_mcht_count_test,site_ad_count_test,site_device_count_test,device_id_count_test,
               device_offer_count_test,merchant_device_count_test,category_ad_count_test,browser_ad_count_test,country_ad_count_test,country_offer_count_test,
               country_site_count_test,country_browser_count_test,device_browser_count_test]

print("merging new features")
for x in agg_df:
    train = train.merge(x)

for x in agg_df_test:
    test = test.merge(x)
print ('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

# Label Encoding
print("Label Encoding features")
from sklearn.preprocessing import LabelEncoder

for c in list(train.select_dtypes(include=['object']).columns):
    if c != 'ID':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

print("calculating feature importance")

# Compute the correlation matrix
corr = train.corr()

print("plotting correlation")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(1100, 1000))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns_plot=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
# sns_plot.savefig("feature_importance_correlation.png")

print("done")
