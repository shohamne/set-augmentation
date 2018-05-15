import train
import csv
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool, cpu_count


with open(train.args.result_file, 'w') as fp:
    writer = csv.DictWriter(fp, train.results.keys())
    writer.writeheader()

train.args.no_cuda = True
train.args.epochs = 500

tasks = []
for seed in range(100,105):
    for max_set_new_size_train in [10, 100]:
        for max_set_new_size_test in [10, 100]:
            for new_sets_number_train in [int(np.ceil(100.0/max_set_new_size_train))]:
                for new_sets_number_test in [int(np.ceil(100.0 / max_set_new_size_test))]:
                    new_args = deepcopy(train.args)
                    new_args.max_set_new_size_train = max_set_new_size_train
                    new_args.new_sets_number_train = new_sets_number_train
                    new_args.max_set_new_size_test = max_set_new_size_test
                    new_args.new_sets_number_test = new_sets_number_test
                    new_args.seed = seed
                    tasks.append(new_args)

pool = Pool(cpu_count())
#pool = Pool(1)

result = pool.map(train.main, tasks)
result_df = pd.DataFrame(result)
pool.close()
pool.join()

result_df.to_csv('{}.csv'.format(train.args.result_file))

