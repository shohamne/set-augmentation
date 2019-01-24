import pandas as pd
import pylab as plt

df = pd.read_csv('batch_train_mnist-dropout.py.csv')
res = df.pivot_table(values = ['train_accuracy', 'test_accuracy'],
               index = 'dropout_ratio',columns = 'network_width_factor')

res.to_csv('mat.csv')

df['one'] = 1
res = df.pivot_table(values = 'data_set', index = 'dropout_ratio',
               columns = 'network_width_factor', aggfunc='count')
res.to_csv('count.csv')