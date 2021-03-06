from os import path, mkdir
import shutil
import train
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, cpu_count

train.args.no_cuda = True
train.args.batch_size = 512
train.args.epochs = 1000
train.args.result_file = path.join('results', '{}.csv'.format(path.basename(__file__)))
train.args.data_set = 'mnist'
train.args.lr = 0.1
train.args.real_epochs_in_epoch = 20

#nproc = cpu_count()
nproc = 1

if not path.exists('results'):
    mkdir('results')

if path.exists('logs'):
    shutil.rmtree('logs')
mkdir('logs')


train.write_csv_header()

tasks = []

task_id = -1

for seed in range(103, 105):
    for network_width_factor in [1, 2, 4]:
        for set_size in [500]:
            for dropout in [0.05, 0.1, 0.2, 0.5, 1.0]:
                task_id += 1
                new_args = deepcopy(train.args)
                new_args.id = task_id
                new_args.max_set_new_size_train = set_size
                new_args.new_sets_number_train = 1
                new_args.max_set_new_size_test = set_size
                new_args.new_sets_number_test = 1
                new_args.set_size_range_train = [set_size, set_size]
                new_args.set_size_range_test = [set_size, set_size]
                new_args.dropout_ratio = dropout
                new_args.network_width_factor = network_width_factor
                new_args.seed = seed
                tasks.append(new_args)

tasks_df = pd.DataFrame([args.__dict__ for args in tasks ]).set_index('id').sort_index()
tasks_df.to_csv('{}.tasks.csv'.format(train.args.result_file))

if nproc > 1:
    pool = Pool(nproc)

    result = pool.map(train.main, tasks)

    pool.close()
    pool.join()
else:
    result = [train.main(args) for args in tasks]

result_df = pd.DataFrame(result)
result_df.to_csv('{}.csv'.format(train.args.result_file))

