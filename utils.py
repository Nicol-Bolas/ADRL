import pandas as pd
import csv
import torch
import shutil
import os,sys,random,datetime
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Args(object):
    
    def __init__(self): 
        self.use_cuda = True
        self.epochs = 100
        self.num_classes = 67336
        self.batch_size = 128
        self.lr = 0.1
        self.num_workers = 4
        self.arch_type = 'visage'
        

def save_epoch(args, model):
    model_state = model.state_dict()
    for key in model_state: model_state[key] = model_state[key].clone().cpu()
    
    checkpoint = '/home/yezilong/my_model/AAA/result/checkpoint/{}_{}.pth.tar'.format(args.arch_type, args.epoch)
    torch.save({
        #'optim_state_dict': optim.state_dict(),
        'model_state_dict': model_state,
    }, checkpoint)
    args.checkpoint = checkpoint        
        
        
def save_checkpoint2(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')        
        
def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()
    
def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def adjust_learning_rate(optimizer, decay_rate=.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
        

        
## K cross validate

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        d = d.strip('\n').split()
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def test_acc(folds, thresholds, predicts, remark):
    accuracy = []
    thd = []
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, [predicts[i] for i in train])
        accuracy.append(eval_acc(best_thresh, [predicts[i] for i in test]))
        thd.append(best_thresh)
    print('\n {} ACC={:.4f} std={:.4f} thd={:.4f}'.format(remark, np.mean(accuracy), 
                                                 np.std(accuracy), np.mean(thd)))
        
        