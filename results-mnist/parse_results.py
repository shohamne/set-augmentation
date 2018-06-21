import pandas as pd
FNAME =  'batch_train_mnist-1.py.csv'

df = pd.read_csv(FNAME)
df['max_set_new_size_train__new_sets_number_train']=map(lambda i: '{},{}'.format(df.loc[i,'max_set_new_size_train'], df.loc[i,'new_sets_number_train']),df.index)
df['max_set_new_size_test__new_sets_number_test']=map(lambda i: '{},{}'.format(df.loc[i,'max_set_new_size_test'], df.loc[i,'new_sets_number_test']),df.index)
gk = ['max_set_new_size_test__new_sets_number_test', 'max_set_new_size_train__new_sets_number_train']
df.groupby(gk).mean()[['train_accuracy', 'test_accuracy']].unstack('max_set_new_size_test__new_sets_number_test').to_csv('mat3.csv')

