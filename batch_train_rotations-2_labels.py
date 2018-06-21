from os import path, mkdir
import train
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, cpu_count

train.args.no_cuda = True
train.args.epochs = 10
train.args.result_file = path.join('results','{}.csv'.format(path.basename(__file__)))
train.args.data_set = 'rotations'
train.args.results_averaging_window = 1
train.args.labels_number = 2
train.args.train_set_size = 100
train.args.test_set_size = 100
train.args.set_size_range_train = [10000,10000]
train.args.set_size_range_test = [10000,10000]

#nproc = cpu_count()
nproc = 1

if not path.exists('results'):
    mkdir('results')

train.write_csv_header()

tasks = []

task_id = 0
for seed in range(1,10):
    for max_set_new_size_train in [100, 10000]:
        for max_set_new_size_test in [100]:
            for new_sets_number_train in [int(np.ceil(10000.0/max_set_new_size_train))]:
                for new_sets_number_test in [int(np.ceil(10000.0 / max_set_new_size_test))]:
                    task_id += 1
                    new_args = deepcopy(train.args)
                    new_args.id = task_id
                    new_args.max_set_new_size_train = max_set_new_size_train
                    new_args.new_sets_number_train = new_sets_number_train
                    new_args.max_set_new_size_test = max_set_new_size_test
                    new_args.new_sets_number_test = new_sets_number_test
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
    result = []
    for args in tasks:
        r = train.main(args)
        result.append(r)

result_df = pd.DataFrame(result)
result_df.to_csv('{}.csv'.format(train.args.result_file))

