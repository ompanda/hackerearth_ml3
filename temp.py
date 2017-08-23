import pandas as pd


base_input_path = './data/input/'
base_output_path = './data/output/'

train = pd.read_csv(base_input_path+"train.csv")
test = pd.read_csv(base_input_path+"test.csv")

print (train.head())
