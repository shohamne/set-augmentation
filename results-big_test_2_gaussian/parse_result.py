import numpy as np
import pandas as pd
import pylab as plt

plt.close('all')
idx = pd.IndexSlice

df = pd.read_csv('batch_train_rotations-2_labels.py.csv')
df = df.append(pd.read_csv('batch_train_rotations-2_labels.py.1000.csv'))
df = df.append(pd.read_csv('batch_train_rotations-2_labels.py.1024.1000.csv'))
df = df.append(pd.read_csv('batch_train_rotations-2_labels.py.1024.csv'))


res_df = df.pivot_table(values = ['train_accuracy','test_accuracy'],
               index= ['train_set_size', 'max_set_new_size_train'],
               columns= 'max_set_new_size_test')

res_df.to_csv('mat.csv')

for train_or_test  in res_df.columns.levels[0]:
    for train_set_size in res_df.index.levels[0]:
        curr_df = res_df[[train_or_test]].loc[idx[train_set_size, :], :]
        curr_df.index = curr_df.index.droplevel(0)
        curr_df.columns = curr_df.columns.droplevel(0)
        curr_df.index = np.log2(curr_df.index)
        if train_or_test == 'train_accuracy':
            curr_df = curr_df.mean(axis=1)

        graph_name = '{};train_set_size={}'.format(train_or_test, train_set_size)

        plt.figure(graph_name)
        plt.plot(curr_df)
        if train_or_test != 'train_accuracy':
            plt.legend(curr_df.columns, title='sets size in test')
        plt.xlabel('sets size in train')
        plt.ylabel('accuracy')
        plt.xticks(curr_df.index, np.int32(np.exp2(curr_df.index)))
        plt.yticks(np.arange(0.5,1,0.1))
        plt.grid()
        plt.title(graph_name)
        plt.show()

