import pandas as pd
FNAME =  'batch_train_mnist-like_dropout.py.csv'

df = pd.read_csv(FNAME)
df[df.seed >= 103][df.seed < 105]
df['max_set_new_size_train__new_sets_number_train']=map(lambda i: '{},{}'.format(df.loc[i,'max_set_new_size_train'], df.loc[i,'new_sets_number_train']),df.index)
df['max_set_new_size_test__new_sets_number_test']=map(lambda i: '{},{}'.format(df.loc[i,'max_set_new_size_test'], df.loc[i,'new_sets_number_test']),df.index)
gk = ['max_set_new_size_test__new_sets_number_test', 'max_set_new_size_train__new_sets_number_train']
df.groupby(gk).mean()[['train_accuracy', 'test_accuracy']].unstack('max_set_new_size_test__new_sets_number_test').to_csv('mat3.csv')


res = df.pivot_table(values = ['train_accuracy', 'test_accuracy'],
               index = 'max_set_new_size_train__new_sets_number_train',columns = 'network_width_factor')

res.to_csv('mat.csv')

res = df.pivot_table(values = 'data_set', index = 'max_set_new_size_train__new_sets_number_train',
               columns = 'network_width_factor', aggfunc='count')
res.to_csv('count.csv')