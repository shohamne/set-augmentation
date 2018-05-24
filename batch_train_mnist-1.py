from os import path, mkdir
import train
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, cpu_count

train.args.no_cuda = False
train.args.batch_size = 512
train.args.epochs = 100
train.args.result_file = path.join('results','{}.csv'.format(path.basename(__file__)))
train.args.data_set = 'mnist'
train.args.lr = 0.1

#nproc = cpu_count()
nproc = 1

if not path.exists('results'):
    mkdir('results')

with open(train.args.result_file, 'w') as fp:
    writer = csv.DictWriter(fp, train.results.keys())
    writer.writeheader()

tasks = []

id = 0
for seed in range(100,105):
    for set_size_range in [[10,100], [100,1000]]:
        for max_set_new_size_train in set_size_range:
            for max_set_new_size_test in set_size_range:
                for new_sets_number_train in [int(np.ceil(float(set_size_range[1])/max_set_new_size_train))]:
                    for new_sets_number_test in [int(np.ceil(float(set_size_range[1]) / max_set_new_size_test))]:
                        id += 1
                        new_args = deepcopy(train.args)
                        new_args.id = id
                        new_args.max_set_new_size_train = max_set_new_size_train
                        new_args.new_sets_number_train = new_sets_number_train
                        new_args.max_set_new_size_test = max_set_new_size_test
                        new_args.new_sets_number_test = new_sets_number_test
                        new_args.set_size_range = set_size_range
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

