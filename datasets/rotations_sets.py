import numpy as np
from torch.utils.data import Dataset

class RotationsSetsDataset(Dataset):
    def __init__(self, nsamples, nlabels=2, set_size_range=[300,500]):
        self.data = []
        self.target = []
        s = np.array([[1,2],
                      [2,1]])
        for i in range(nsamples):
            alpha = np.random.rand()*np.pi

            M = set_size_range[0] if set_size_range[0] == set_size_range[1] \
                else np.random.randint(set_size_range[0], set_size_range[1])

            R = np.array([[np.cos(alpha),np.sin(alpha)],
                         [-np.sin(alpha),np.cos(alpha)]])

            Sigma = np.matmul(s,s.transpose())
            RSigmaRt = np.matmul(np.matmul(R,Sigma),R.transpose())

            data = np.zeros([set_size_range[1], 2])
            data[:M,:] =  np.random.multivariate_normal([0,0], RSigmaRt, M)

            label = int(alpha/np.pi*nlabels)
            target = np.zeros(nlabels); target[label] = 1

            self.data.append(np.float32(data))
            self.target.append(np.int64(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [self.data[item], self.target[item]]




