from __future__ import print_function
from copy import deepcopy

import time
import argparse
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorchtools import EarlyStopping

from datasets.rotations_sets import RotationsSetsDataset
from datasets.mnist_sets import MnistSetsDataset
from datasets.augmentations import SubsampleDataset

from cStringIO import StringIO
import logging

# Training settings
parser = argparse.ArgumentParser(description='set augmentations test')
parser.add_argument('--id', type=str, default='0', metavar='N',
                    help='just an ID for logging')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--reduce-on-plateau-patience', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--early-stopping-patience', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--dropout_ratio', type=float, default=1.0, metavar='M',
                    help='dropout_ratio (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--results-averaging-window', type=int, default=10, metavar='N',
                    help='how many results at the end to take for averaging')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-set-size', type=int, default=60, metavar='N',
                    help='how many sampels in the train set')
parser.add_argument('--test-set-size', type=int, default=1000, metavar='N',
                    help='how many sampels in the test set')
parser.add_argument('--labels-number', type=int, default=2, metavar='N',
                    help='how many sampels in the test set')
parser.add_argument('--set-size-range-train', type=int, nargs=2, default=(100, 1000), metavar=('min', 'max'),
                    help='how many sampels in the train sets')
parser.add_argument('--set-size-range-test', type=int, nargs=2, default=(100, 1000), metavar=('min', 'max'),
                    help='how many sampels in the train sets')
parser.add_argument('--max-set-new-size-train', type=int, default=100, metavar='N',
                    help='maximum number of sampels in the train set after subsample')
parser.add_argument('--new-sets-number-train', type=int, default=1, metavar='N',
                    help='number of new subsampled sets from each original set')
parser.add_argument('--max-set-new-size-test', type=int, default=100, metavar='N',
                    help='maximum number of sampels in the test set after subsample')
parser.add_argument('--new-sets-number-test', type=int, default=1, metavar='N',
                    help='number of new subsampled sets from each original set')
parser.add_argument('--random-sample', action='store_true', default=False,
                    help='task_id set random sampling of the sets is used')
parser.add_argument('--result-file', type=str, default='/tmp/augmentation_result.csv', metavar='S',
                    help='file name for result')
parser.add_argument('--data-set', choices=['rotations', 'mnist'], type=str, default='mnist', metavar='S',
                    help='choose data set from [mnist,rotations]')
parser.add_argument('--network-with-factor', type=int, default=1, metavar='N',
                    help='factor for the network with')

args = parser.parse_args()

results = args.__dict__
results['train_loss'] = None
results['train_accuracy'] = None
results['test_accuracy'] = None
results['time'] = None


def length(x):
    used = torch.sign(torch.max(torch.abs(x), x.dim() - 1)[0])
    length = torch.sum(used, used.dim() - 1, keepdim=True)
    return length


def write_csv_header():
    with open(args.result_file, 'w') as fp:
        writer = csv.DictWriter(fp, sorted(results.keys()))
        writer.writeheader()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def dropout(data, dropout_ratio):
    assert (0 < dropout_ratio <= 1)
    if dropout_ratio < 1:
        nsamples_orig = data.shape[1]
        nsamples_out = int(nsamples_orig * dropout_ratio)

        inds = np.random.permutation(nsamples_orig)[:nsamples_out]

        data_out = data[:, inds, :]

        return data_out
    else:
        return data


def main(args):
    print('Arguments: {}'.format(args.__dict__))

    logging.basicConfig(
        level=logging.INFO,
    )
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(console)
    file_handler = logging.FileHandler("{}/{}.log".format('logs', args.id))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    for k in args.__dict__.keys():
        results[k] = args.__dict__[k]

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.data_set == 'rotations':
        orig_train_set = RotationsSetsDataset(args.train_set_size, args.labels_number,
                                              set_size_range=args.set_size_range_train)
        orig_test_set = RotationsSetsDataset(args.test_set_size, args.labels_number,
                                             set_size_range=args.set_size_range_test)
        labels_number = args.labels_number
    if args.data_set == 'mnist':
        orig_train_set = MnistSetsDataset(train=True, set_size_range=args.set_size_range_train)
        orig_test_set = MnistSetsDataset(train=False, set_size_range=args.set_size_range_test)
        labels_number = 10

    orig_test_set_sizes = np.array([np.float64(length(torch.Tensor(s))) for s, _ in orig_test_set])

    train_set = SubsampleDataset(orig_train_set, args.max_set_new_size_train, args.new_sets_number_train,
                                 args.random_sample)
    test_set = SubsampleDataset(orig_test_set, args.max_set_new_size_test, args.new_sets_number_test,
                                args.random_sample)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            w = args.network_with_factor
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 100*w)  # , bias=False)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            self.fc2 = nn.Linear(100*w, 50*w)  # , bias=False)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            self.fc3 = nn.Linear(50*w, 50*w)  # , bias=False)
            torch.nn.init.xavier_uniform(self.fc3.weight)

            self.fc4 = nn.Linear(50*w, 50*w)
            torch.nn.init.xavier_uniform(self.fc4.weight)
            self.fc4_bn = nn.BatchNorm1d(50*w)
            self.fc5 = nn.Linear(50*w, 50*w)
            torch.nn.init.xavier_uniform(self.fc5.weight)
            self.fc5_bn = nn.BatchNorm1d(50*w)
            self.fc6 = nn.Linear(50*w, labels_number)
            torch.nn.init.xavier_normal(self.fc6.weight)

        def forward(self, x):
            x = F.dropout(F.relu((self.fc1(x))), training=False)  # #self.training)
            x = F.dropout(F.relu((self.fc2(x))), training=False)  # #self.training)
            x = F.dropout(F.relu((self.fc3(x))), training=False)  # #self.training)

            x = torch.div(torch.sum(x, dim=1), length(x))
            # x = torch.sum(x, dim=1)

            x = F.dropout(F.relu((self.fc4_bn(self.fc4(x)))), training=False)  # self.training))False)#
            x = F.dropout(F.relu((self.fc5_bn(self.fc5(x)))), training=False)  # self.training)
            x = self.fc6(x)

            y = F.softmax(x, 1)

            return y

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.reduce_on_plateau_patience,
                                                     verbose=True)
    early_stopping = EarlyStopping(verbose=True, patience=args.early_stopping_patience)

    def train(epoch, optimizer):
        model.train()
        losses = []
        accuracies = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = dropout(data, args.dropout_ratio)
            data, target = data.to(device), target.to(device)
            if data.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            correct = output.topk(1)[1].reshape(target.shape) == target
            accuracy = np.mean(np.float32(correct.cpu()))
            loss.backward()
            optimizer.step()
            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}\tLR: {}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item(), accuracy, lr))
            losses.append(loss.item())
            accuracies.append(accuracy)

        mean_losses = np.mean(losses)
        mean_accuracy = np.mean(accuracies)

        print('Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.2f}\tLR: {}'.format(
            epoch, np.mean(mean_losses), np.mean(mean_accuracy), get_lr(optimizer)),
            end="", file=log_sstream)

        return mean_losses, mean_accuracy

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        out = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                out.append(np.array(output.cpu()))

            print('\tTest set: Loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)),
                end="", file=log_sstream)

        out = np.concatenate(out)
        score = out * (out == out.max(axis=1, keepdims=True))
        score_df = pd.concat([
            pd.DataFrame({'orig_ind': test_set.orig_inds, 'target': test_set.target}),
            pd.DataFrame(score)], axis=1)

        group_score = score_df.groupby(by='orig_ind')[range(score.shape[1])].sum()
        group_target = score_df.groupby(by='orig_ind')['target'].unique()
        group_pred = group_score.idxmax(axis=1)  # get the index of the max log-probability
        group_correct = group_target == group_pred
        n_group_correct = float(group_correct.sum())
        accuracy = n_group_correct / len(group_target)

        print('\tGroup Test: Accuracy: {}/{} ({:.0f}%)'.format(
            n_group_correct, len(group_target),
            100. * accuracy), end="", file=log_sstream)

        # import pylab as plt
        # plt.scatter(orig_test_set_sizes, group_correct); plt.show()

        return accuracy

    # import numpy as np
    # import pylab as plt
    # plt.close('all')
    # cmap = np.random.rand(args.labels_number,3)
    # for d,t in train_loader:
    #     for dd,tt in zip(d,t):
    #         z = np.array(dd)
    #         plt.sscatter(z[:, 0], z[:, 1], color=cmap[np.array(tt)])
    #         plt.hold('on')
    #         #z = z.sum(axis=0)
    #         #plt.scatter(z[0], z[1], color=['r', 'b', 'g', 'c'][np.array(tt)],marker=1)
    # plt.show()

    lr = args.lr
    test_accuracies = []
    train_losses = []
    train_accuracies = []
    t0 = t2 = time.time()
    for epoch in range(1, args.epochs + 1):
        log_sstream = StringIO()
        print('ID: {}\t'.format(args.id), end="", file=log_sstream)
        train_loss, train_accuracy = train(epoch, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # lr = exp_lr_scheduler(epoch, init_lr=args.lr, lr_decay_epoch=args.lr_decay_epoch)

        test_accuracy = test()
        scheduler.step(train_loss)
        test_accuracies.append(test_accuracy)
        t1 = time.time()
        print('\tTime: {:.0f} Epoch Time: {:.0f}'.format(t1 - t0, t1 - t2), file=log_sstream)
        logger.info(log_sstream.getvalue())
        t2 = time.time()
        if get_lr(optimizer) < lr*0.0001:
            early_stopping(-test_accuracy)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    results['time'] = t2 - t0
    results['train_loss'] = np.mean(train_losses[-args.results_averaging_window:])
    results['train_accuracy'] = np.mean(train_accuracies[-args.results_averaging_window:])
    results['test_accuracy'] = np.mean(test_accuracies[-args.results_averaging_window:])

    print('Final Test Accuracy: {:.2f}'.format(results['test_accuracy']))

    with open(args.result_file, 'a') as fp:
        writer = csv.DictWriter(fp, sorted(results.keys()))
        writer.writerow(results)

    return deepcopy(results)


if __name__ == '__main__':
    main(args)
