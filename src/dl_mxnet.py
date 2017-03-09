'''
average: about 95
highest: 95.6140
a little improvement from v4
'''
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import datetime
import gzip
import os
import datetime

LEN_DIM = 7
ENGY_THRES = 0.2
#BATCH_NUM = len(train_data)

def read_file(f_name):
    f = open(f_name, 'r')
    i = -1
    info = [[],[]]
    data_raw = []
    label_raw = []
    name_raw = []
    for line in f:
        i = i+1
        if (i % 6 == 0 or i % 6 == 5):
            continue
        elif (i % 6 == 1):
            info[0] = map(float, line.split(' '))
        elif (i % 6 == 2):
            info[1] = map(float, line.split(' '))
            st = 0
            ed = len(info[0])
            
            data_i = [[], []]
            flag = False
            for t in xrange(len(info[0])):
                if (info[0][t]*info[1][t] > 1 or flag is True):
                    #flag = True
                    data_i[0].append(info[0][t])
                    data_i[1].append(info[1][t])
            for t in reversed(xrange(len(data_i[0]))):
                if (data_i[0][t] *data_i[1][t] < 1):
                    data_i[0].pop()
                    data_i[1].pop()
                else:
                    break
            data_i = np.array(data_i)
            #print data_i.shape
            data_raw.append(data_i)
        elif (i % 6 == 3):
            label_raw.append(int(line))
        elif (i % 6 == 4):
            name_raw.append(line)
    
    return data_raw, label_raw, name_raw

def preprocess(data_raw):
    # engy
    data_ret = []
    for i in xrange(len(data_raw)):
        data_i = data_raw[i]
        length = len(data_i[0])
        avg_engy = np.mean(data_i[0])
        data_o = [[], []]
        flag = False
        for t in xrange(length):
            if (flag == False and data_i[0][t] < ENGY_THRES * avg_engy):
                pass
            else:
                flag = True
                data_o[0].append(data_i[0][t])
                data_o[1].append(data_i[1][t])
        length = len(data_o[0])
        for t in reversed(xrange(length)):
            if (data_o[0][t] < ENGY_THRES * avg_engy):
                data_o[0].pop()
                data_o[1].pop()
            else:
                break
        len1 = len(data_o[0])
        len2 = len(data_o[1])
        data_o[0] = data_o[0][len1 / 100 : len1]
        data_o[1] = data_o[1][len2 / 100 : len2]
        data_ret.append(np.array(data_o))
    # f
    # what if we only use dp instead of NN?
    mean = [-0.00368890111513, 0.00842593986922, 0.00261934419705, -0.0325935548158]
    std = [0.0502888828252, 0.0501864781776, 0.0358012690812, 0.0630625440446]
    for i in xrange(len(data_ret)):
        dp = [[0] * len(data_ret[i][1])] * 4
        pre = [[-1] * len(data_ret[i][1])] * 4
        mx = 0
        pos = 0
        posT = 0
        for j in xrange(len(data_ret[i][1])):
            y = np.log(data_ret[i][1][j])
            for t in xrange(4):
                for k in reversed(xrange(j)):
                    x = np.log(data_ret[i][1][k])
                    s = np.sqrt(std[t] * std[t] * (j - k))
                    if (mean[t] - s - s < y - x and y - x < mean[t] + s + s):
                        dp[t][j] = dp[t][k] + 1
                        pre[t][j] = k
                        if (mx < dp[t][j]):
                            mx = dp[t][j]
                            pos = j
                            posT = t
                        break
        data_tmp = [[], []]
        while (pos != -1):
            data_tmp[0].append(data_ret[i][0][pos])
            data_tmp[1].append(data_ret[i][1][pos])
            pos = pre[posT][pos]
        data_ret[i] = [list(reversed(data_tmp[0])), list(reversed(data_tmp[1]))]
    return data_ret

def gap_data_process(data_raw, label_raw):
    M = len(data_raw)
    X = np.zeros((M, 2*LEN_DIM))
    y = np.zeros((M, 4))
    
    for b_n in range(M):
        data_i = data_raw[b_n]
        gap = len(data_i[0]) / LEN_DIM
        for t in range(LEN_DIM):
            X[b_n][t] = data_i[0][t*gap + gap/2]
            X[b_n][t+LEN_DIM] = data_i[1][t*gap + gap/2]
    
        y[b_n][label_raw[b_n]-1] = 1
    return X, y

def gap_diff_data_process(data_raw, label_raw):
    M = len(data_raw)
    X = np.zeros((M, 2*LEN_DIM-1))
    y = np.zeros((M, 4))
    
    for b_n in range(M):
        data_i = data_raw[b_n]
        gap = len(data_i[0]) / LEN_DIM
        for t in range(LEN_DIM):
            X[b_n][t] = np.mean(data_i[0][t*gap:(t+1)*gap])
        for t in range(LEN_DIM-1):
            X[b_n][t+LEN_DIM] = np.mean(data_i[1][(t+1)*gap:(t+2)*gap]) - np.mean(data_i[1][t*gap:(t+1)*gap])

        y[b_n][label_raw[b_n]-1] = 1
    
    return X, y

def gap_slope_data_process(data_raw, label_raw):
    
    def residuals(p, x, y):
        a, b = p
        return (a*x+b) - y
    
    M = len(data_raw)
    X = np.zeros((M, 2*LEN_DIM))
    y = np.zeros((M, 4))
    for b_n in range(M):
        #print label_raw[b_n]
        data_i = data_raw[b_n]
        gap = len(data_i[0]) / LEN_DIM
        for t in range(LEN_DIM):
            y1 = np.array(data_i[1][t*gap:(t+1)*gap])
            y1 = (y1 - np.mean(y1)) #/ np.std(y1)
            x1 = np.array(range(gap))
            #print x1
            #print y1
            r = leastsq(residuals, [1,1], args=(x1, y1))
            X[b_n][t] = r[0][0]
            #X[b_n][t+LEN_DIM] = (np.mean(y1) - np.mean(data_i[1])) / np.sqrt(np.var(data_i[1]))
            X[b_n][t+LEN_DIM] = np.mean(y1)
            #X[b_n][t+LEN_DIM+LEN_DIM] = np.var(y1 - residuals(r[0], x1, y1))
            #print X[b_n][t], X[b_n][t+LEN_DIM]
        
        y[b_n][label_raw[b_n]-1] = 1
    
    return X, y


def data_process(data_raw, label_raw):
    X, y = gap_slope_data_process(data_raw, label_raw)
    #X2, y2 = gap_diff_data_process(data_raw, label_raw)
    #X = np.r_['1', X1, X2]
    #y = np.r_['1', y1, y2]
    y0 = np.zeros([y.shape[0]])
    for i in xrange(y.shape[0]):
        for j in xrange(4):
            if (y[i][j] > 0.9):
                y0[i] = j
    return X, y0

raw_train_data, train_label, train_name = read_file('data/train.txt')
raw_test1_data, test1_label, test1_name = read_file('data/test.txt')
raw_test2_data, test2_label, test2_name = read_file('data/test_new.txt')

train_data = preprocess(raw_train_data)
X_train, y_train = data_process(train_data, train_label)
print X_train.shape, y_train.shape


test1_data = preprocess(raw_test1_data)
X_test1, y_test1 = data_process(test1_data, test1_label)
print X_test1.shape, y_test1.shape

test2_data = preprocess(raw_test2_data)
X_test2, y_test2 = data_process(test2_data, test2_label)
print X_test2.shape, y_test2.shape
HIDDEN_DIM = [4]
CLASS = 4
DEPTH = 0 ## doesn't consider the first and final hidden layer
LEARNING_RATE = 1e-3
REG = 1e-1
X_DIM = len(X_train[0])

TcmpS = datetime.datetime.now()

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc0', num_hidden=HIDDEN_DIM[0])
#net = mx.sym.BatchNorm(net, name='bn0')
net = mx.sym.Activation(net, name='relu0', act_type='relu')
for i in xrange(DEPTH):
    net = mx.sym.FullyConnected(net, name='fc%d' % (i + 1), num_hidden=HIDDEN_DIM[i+1])
    net = mx.sym.Activation(net, name='relu%d' % (i + 1), act_type='relu')
    #net = mx.sym.BatchNorm(net, name='bn%d' % (i + 1))
net = mx.sym.FullyConnected(net, name='relu%d' % (DEPTH + 1), num_hidden = CLASS)
net = mx.sym.SoftmaxOutput(net, name='softmax')

shape = {"data": (40, X_train.shape[1])}
mx.viz.plot_network(symbol=net, shape=shape)

import logging
logging.getLogger().setLevel(logging.DEBUG)

EPOCH = 1000
batch_size=40

model = mx.model.FeedForward(
    symbol=net,
    ctx=mx.cpu(),
    optimizer='adam',
    learning_rate=LEARNING_RATE,
    wd=REG,
    num_epoch=EPOCH
)

TcmpE = datetime.datetime.now()
train = mx.io.NDArrayIter(X_train, y_train, batch_size=batch_size, shuffle = True)
#test1 = mx.io.NDArrayIter(X_test1, y_test1, batch_size=batch_size)#X_test1.shape[0])
test2 = mx.io.NDArrayIter(X_test2, y_test2, batch_size=batch_size)#, shuffle=True)#X_test2.shape[0])

TtrainS = datetime.datetime.now()
model.fit(
    X=train, eval_data=test2,
    batch_end_callback = mx.callback.Speedometer(batch_size, 10)
)

prob = model.predict(X_test2)
res = 0.0;
for i in xrange(X_test2.shape[0]):
    t = prob[i].argmax()
    if (t == y_test2[i]):
        res += 1.0
    else:
        print "Wrong prediction"
        print "raw:\n", raw_test2_data[i]
        print "after preprocess:\n", test2_data[i]
        print "prob:\n", prob[i]
        print "label:", test2_label[i]
        print "name:", test2_name[i]
        
TtrainE = datetime.datetime.now()
print "val_acc=%f" % (res / X_test2.shape[0])
print ("Compiling time: {}".format(TcmpE - TcmpS))
print ("Training {} epoch time: {}".format(EPOCH, TtrainE - TtrainS))
