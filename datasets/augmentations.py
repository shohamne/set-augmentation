import numpy as np
from torch.utils.data import Dataset

def _remove_set_zeros(s):
    return s[np.abs(s).sum(axis=1) > 0,:]

def _randperm(size, n_rand_perms):
    randperm = np.concatenate([np.random.permutation(size) for i in range(n_rand_perms)])
    return randperm

def set_subsample(s, max_set_new_size, new_sets_number):
    orig_set = _remove_set_zeros(s)
    orig_set_size = len(orig_set)
    n_new_sets = new_sets_number if new_sets_number > 0 \
        else int(np.ceil(float(orig_set_size) / max_set_new_size))
    n_rand_perms = int(np.ceil(float(n_new_sets)*max_set_new_size/orig_set_size))
    randperm = _randperm(orig_set_size, n_rand_perms)
    new_sets = []
    for i in range(0, n_new_sets*max_set_new_size, max_set_new_size):
        new_set = orig_set[randperm[i:i+max_set_new_size]]
        padded = np.zeros([max_set_new_size, s.shape[1]], orig_set.dtype)
        padded[:new_set.shape[0], :] = new_set
        new_sets.append(padded)
    return new_sets


def set_random_subsample(s, max_set_new_size, new_sets_number):
    orig_set = _remove_set_zeros(s)
    orig_set_size = len(orig_set)
    new_sets = []
    for i in range(0, new_sets_number):
        randperm = np.random.permutation(orig_set_size)
        new_set = orig_set[randperm[:min(max_set_new_size, orig_set_size)]]
        padded = np.zeros([max_set_new_size, s.shape[1]], orig_set.dtype)
        padded[:new_set.shape[0], :] = new_set
        new_sets.append(padded)
    return new_sets


class SubsampleDataset(Dataset):
    def __init__(self, sets_dataset, max_set_new_size, new_sets_number, random_sample):
        new_data = []
        new_target = []
        orig_inds = []
        for i in range(len(sets_dataset)):
            data, target = sets_dataset[i]
            subsampled_data = set_subsample(data, max_set_new_size, new_sets_number) \
                if not random_sample else set_random_subsample(data, max_set_new_size, new_sets_number)
            new_data += subsampled_data
            new_target += [target]*len(subsampled_data)
            orig_inds += [i]*len(subsampled_data)


        self.data = new_data
        self.target = new_target
        self.orig_inds = orig_inds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [self.data[item], self.target[item]]



