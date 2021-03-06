from os import path, mkdir
import train
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, cpu_count


S = 1024
SET_NEW_SIZES = np.int32(np.exp2(np.arange(0, np.log2(S)+1)))

train.args.no_cuda = True
train.args.epochs = 10
train.args.result_file = path.join('results','{}.csv'.format(path.basename(__file__)))
train.args.data_set = 'rotations'
train.args.results_averaging_window = 1
train.args.labels_number = 2
train.args.test_set_size = 100
train.args.set_size_range_train = [S,S]

nproc = cpu_count()
#nproc = 1

if not path.exists('results'):
    mkdir('results')

train.write_csv_header()

tasks = []

task_id = 0
for seed in range(1,10):
    for train_set_size in [10, 100]:
        for max_set_new_size_train in SET_NEW_SIZES:
            for max_set_new_size_test in SET_NEW_SIZES:
                set_size_range_test = [S, S]
                new_sets_number_train = int(np.ceil(float(S)/max_set_new_size_train))
                new_sets_number_test = int(np.ceil(set_size_range_test[0]/max_set_new_size_test))

                task_id += 1
                new_args = deepcopy(train.args)
                new_args.id = task_id
                new_args.train_set_size = train_set_size
                new_args.set_size_range_test = set_size_range_test
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

