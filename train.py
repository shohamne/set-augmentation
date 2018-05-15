from __future__ import print_function
import argparse
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets.rotations_sets import RotationsSetsDataset
from datasets.augmentations import SubsampleDataset

# Training settings
parser = argparse.ArgumentParser(description='set augmentations test')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr-decay-epoch', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                    help='learning rate (default: 0.000001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-set-size', type=int, default=60, metavar='N',
                    help='how many sampels in the train set')
parser.add_argument('--test-set-size', type=int, default=1000, metavar='N',
                    help='how many sampels in the test set')
parser.add_argument('--labels-number', type=int, default=4, metavar='N',
                    help='how many sampels in the test set')
parser.add_argument('--set-size-range', type=int, nargs=2, default=(10,100), metavar=('min','max'),
                    help='how many sampels in the test set')
parser.add_argument('--max-set-new-size-train', type=int, default=60, metavar='N',
                    help='maximum number of sampels in the train set after subsample')
parser.add_argument('--new-sets-number-train', type=int, default=2, metavar='N',
                    help='number of new subsampled sets from each original set')
parser.add_argument('--max-set-new-size-test', type=int, default=60, metavar='N',
                    help='maximum number of sampels in the test set after subsample')
parser.add_argument('--new-sets-number-test', type=int, default=2, metavar='N',
                    help='number of new subsampled sets from each original set')
parser.add_argument('--random-sample', action='store_true', default=False,
                    help='id set random sampling of the sets is used')
parser.add_argument('--result-file', type=str, default='/tmp/augmentation_result.csv', metavar='S',
                    help='file name for result')

args = parser.parse_args()

results =  args.__dict__
results['test_accuracy'] = None


def main(args):
    print ('Arguments: {}'.format(args.__dict__) )

    for k in args.__dict__.keys():
        results[k]=args.__dict__[k]

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    orig_train_set = RotationsSetsDataset(args.train_set_size, args.labels_number, set_size_range=args.set_size_range)
    orig_test_set = RotationsSetsDataset(args.test_set_size, args.labels_number, set_size_range=args.set_size_range)

    train_set = SubsampleDataset(orig_train_set, args.max_set_new_size_train, args.new_sets_number_train, args.random_sample)
    test_set = SubsampleDataset(orig_test_set, args.max_set_new_size_test, args.new_sets_number_test, args.random_sample)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    def length(x):
        used = torch.sign(torch.max(torch.abs(x), x.dim()-1)[0])
        length = torch.sum(used, 1, keepdim=True)
        return length


    def exp_lr_scheduler(epoch, init_lr, lr_decay_epoch):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))
        return lr


    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 100)#, bias=False)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            self.fc2 = nn.Linear(100, 50)#, bias=False)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            self.fc3 = nn.Linear(50, 50)#, bias=False)
            torch.nn.init.xavier_uniform(self.fc3.weight)

            self.fc4 = nn.Linear(50, 50)
            torch.nn.init.xavier_uniform(self.fc4.weight)
            self.fc4_bn = nn.BatchNorm1d(50)
            self.fc5 = nn.Linear(50, 50)
            torch.nn.init.xavier_uniform(self.fc5.weight)
            self.fc5_bn = nn.BatchNorm1d(50)
            self.fc6 = nn.Linear(50, args.labels_number)
            torch.nn.init.xavier_normal(self.fc6.weight)


        def forward(self, x):
            x = F.dropout(F.relu((self.fc1(x))), training=False)# #self.training)
            x = F.dropout(F.relu((self.fc2(x))), training=False)# #self.training)
            x = F.dropout(F.relu((self.fc3(x))), training=False)# #self.training)

            x = torch.div(torch.sum(x, dim=1),length(x))
            #x = torch.sum(x, dim=1)

            x = F.dropout(F.relu((self.fc4_bn(self.fc4(x)))), training=False)#self.training))False)#
            x = F.dropout(F.relu((self.fc5_bn(self.fc5(x)))), training=False)#self.training)
            x = self.fc6(x)

            y = F.softmax(x,1)

            return y

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch, optimizer, lr):
        set_lr(optimizer, lr)
        model.train()
        losses = []
        accuracies = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if data.shape[0] <= 1:
                continue
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            correct = output.topk(1)[1].reshape(target.shape) == target
            accuracy = np.mean(np.float32(correct))
            loss.backward()
            optimizer.step()
            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}\tLR: {}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item(), accuracy, lr))
            losses.append(loss.item())
            accuracies.append(accuracy)
        print('Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.2f}\tLR: {}'.format(
            epoch, np.mean(losses), np.mean(accuracies), lr),
            end="")

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        out = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                out.append(np.array(output))

            print('\t\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)),
                end="")

        out = np.concatenate(out)
        score = out * (out == out.max(axis=1, keepdims=True))
        score_df = pd.concat([
            pd.DataFrame({'orig_ind': test_set.orig_inds, 'target': test_set.target}),
            pd.DataFrame(score)],axis=1)

        group_score = score_df.groupby(by='orig_ind')[range(score.shape[1])].sum()
        group_target = score_df.groupby(by='orig_ind')['target'].unique()
        group_pred = group_score.idxmax(axis=1)  # get the index of the max log-probability
        group_correct = (group_target==group_pred).sum()
        accuracy = float(group_correct) / len(group_target)

        print('\t\tGroup Test: Accuracy: {}/{} ({:.0f}%)'.format(
            group_correct, len(group_target),
            100. * accuracy))

        return accuracy



    # import numpy as np
    # import pylab as plt
    # plt.close('all')
    # cmap = np.random.rand(args.labels_number,3)
    # for d,t in train_loader:
    #     for dd,tt in zip(d,t):
    #         z = np.array(dd)
    #         plt.scatter(z[:, 0], z[:, 1], color=cmap[np.array(tt)])
    #         plt.hold('on')
    #         #z = z.sum(axis=0)
    #         #plt.scatter(z[0], z[1], color=['r', 'b', 'g', 'c'][np.array(tt)],marker=1)
    # plt.show()

    lr = args.lr
    test_accuracies = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, optimizer, lr)
        lr = exp_lr_scheduler(epoch, init_lr=args.lr, lr_decay_epoch=args.lr_decay_epoch)
        test_accuracy = test()
        test_accuracies.append(test_accuracy)

    results['test_accuracy'] = np.mean(test_accuracies[-10:])

    print ('Final Test Accuracy: {:.2f}'.format(results['test_accuracy']) )

    with open(args.result_file, 'a') as fp:
        writer = csv.DictWriter(fp, results.keys())
        writer.writerow(results)

    return results



if __name__ == '__main__':
    main(args)