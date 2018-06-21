import pandas as pd
import pylab as plt

df = pd.read_csv('batch_train_rotations-2_labels.py.csv')
res = df.pivot_table(values = ['train_accuracy','test_accuracy'],
               index= 'max_set_new_size_train',
               columns= 'max_set_new_size_test')

res.to_csv('mat.csv')