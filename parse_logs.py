import pandas as pd
import pylab as plt
import glob

LOG_ID = '0'

def parse_line(line):
    train_line, test_line  = line.split('Test set: ')
    train_list = train_line.replace('Train ', '').replace(' ', '\t').split('\t')
    test_list = test_line.replace('Train ', '').replace(',','\t').replace(' ', '\t').split('\t')

    ret = dict()
    ret['id'] = train_list[train_list.index('ID:')+1]
    ret['epoch'] = int(train_list[train_list.index('Epoch:')+1])

    if   ret['id']=='2' and  ret['epoch']==1:
        zz=1
    ret['train_loss'] = float(train_list[train_list.index('Loss:')+1])
    ret['train_accuracy'] = float(train_list[train_list.index('Accuracy:')+1])

    ret['test_loss'] =  float(test_list[test_list.index('Loss:')+1])
    test_accuracy_ratio = test_list[test_list.index('Accuracy:')+1].split('/')
    ret['test_accuracy'] = float(test_accuracy_ratio[0])/float(test_accuracy_ratio[1])

    return ret

content = []
for fname in glob.glob('logs/*'):
    with open(fname) as f:
        content += f.readlines()

content = set([x.strip() for x in content])

df = pd.DataFrame([parse_line(l) for l in content if len(l)>0])
df = df[df['id']==LOG_ID]
df = df.set_index('epoch').sort_index()

#plt.figure('loss')
df[['train_loss', 'test_loss']].plot()

#plt.figure('accuracy')
df[['train_accuracy', 'test_accuracy']].plot()

plt.show()





