import csv
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
import numpy as np

base_input_path = './data/input/'
base_output_path = './data/output/'

print("reading files")
# read datasets
#leaderborad_score - .67818
# df_xgb = pd.read_csv(base_output_path+'adclick_xgb-v3-auc-0.985677862407.csv')
# df_stacked = pd.read_csv(base_output_path+'ensemble-v1-auc-0.972098664126.csv')

#leaderborad_score - .67838
# df_xgb = pd.read_csv(base_output_path+'adclick_xgb-v3-auc-0.985308658497.csv')
# df_stacked = pd.read_csv(base_output_path+'ensemble-v1-auc-0.972098664126.csv')


df_xgb = pd.read_csv(base_output_path+'adclick_xgb-v4-auc-0.989853672733.csv')
df_stacked = pd.read_csv(base_output_path+'stacking-v1-auc-0.971802729873.csv')

#merge and ensemble
y_pred_xgb= df_xgb.click;
y_pred_stacked = df_stacked.click

print("calculating average precision")
# y_pred_final=y_pred_stacked * 0.2855 + y_pred_xgb * 0.7145
y_pred_final=y_pred_xgb#y_pred_stacked * 0.1855 + y_pred_xgb * 0.8145

print ("Preparing submit file")

output = pd.DataFrame({'ID': df_xgb['ID'], 'click': y_pred_final})
output.to_csv(base_output_path+'merge_v_{}.csv'.format(12), index=False)

print("done")

