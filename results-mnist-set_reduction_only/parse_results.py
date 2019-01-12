import pandas as pd
FNAME =  'batch_train_mnist-set_reduction_only.py.csv'

df = pd.read_csv(FNAME)
df['set_new_size_train']=df['max_set_new_size_train']
df['set_orig_size_test']=df['max_set_new_size_test']
gk = ['set_orig_size_test', 'set_new_size_train']
df.groupby(gk).mean()[['train_accuracy', 'test_accuracy']].unstack('set_orig_size_test').to_csv('mat3.csv')

