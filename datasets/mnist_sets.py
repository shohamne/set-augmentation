import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

_ROWS28x28, _COLS28x28 = np.where(np.ones((28,28)))

def _to01(x):
    m = x.min()
    M = x.max()
    return (x-m)/(M-m)

def _pixels_choise(img, npixels):

    p = img.flatten() / img.sum()
    inds = np.where(p)[0]
    p = p[inds]

    choise = np.array(np.random.choice(inds, size=npixels, p=p))

    rows = _ROWS28x28[choise]
    cols = _COLS28x28[choise]

    return np.array(zip(rows,cols))


def _img2set(img, set_size_range=[300,500]):
    set_size = set_size_range[0] if set_size_range[0] == set_size_range[1] \
        else np.random.randint(set_size_range[0], set_size_range[1])

    s = _pixels_choise(img.squeeze(), set_size)
    padded = np.zeros((set_size_range[1], 2),dtype=np.float32)
    padded[:s.shape[0],:] = s

    return padded






class MnistSetsDataset(Dataset):
    def __init__(self, train, set_size_range=[300,500]):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,)),
                                    transforms.Lambda(lambda x: _to01(x)),
                                    transforms.Lambda(lambda x: np.array(x)),
                                    transforms.Lambda(lambda x: _img2set(x, set_size_range)),
                                    ])
        # if not exist, download mnist dataset
        #train_set = MNIST(root=root, train=True, transform=trans, download=True)
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        mnist_dataset = MNIST(root=root, train=train, transform=trans, download=True)

        self._dataset = [(data, target) for data, target in mnist_dataset]


    def __getitem__(self, item):
        return self._dataset[item]

    def __len__(self):
        return len(self._dataset)

if __name__ == '__main__':
    import pylab as plt
    dataset = MnistSetsDataset(train=True, set_size_range=[10,100])

    for i in range(10):
        data=dataset[i][0]
        im = np.zeros([28,28])
        for row,col in data:
            im[int(row),int(col)] += 1

        plt.imshow(im)
        plt.show()


