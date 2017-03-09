
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import datetime
from cntk import Trainer, learning_rate_schedule, UnitType, momentum_schedule
from cntk.device import cpu, set_default_device
from cntk.learner import *
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss


# In[2]:

LEN_DIM = 7
ENGY_THRES = 0.2


# In[3]:

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
            info[0] = list(map(float, line.split(' ')))
        elif (i % 6 == 2):
            info[1] = list(map(float, line.split(' ')))
            st = 0
            ed = len(info[0])
            
            data_i = [[], []]
            flag = False
            for t in range(len(info[0])):
                if (info[0][t]*info[1][t] > 1 or flag is True):
                    #flag = True
                    data_i[0].append(info[0][t])
                    data_i[1].append(info[1][t])
            for t in reversed(range(len(data_i[0]))):
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
    for i in range(len(data_raw)):
        data_i = data_raw[i]
        length = len(data_i[0])
        avg_engy = np.mean(data_i[0])
        data_o = [[], []]
        flag = False
        for t in range(length):
            if (flag == False and data_i[0][t] < ENGY_THRES * avg_engy):
                pass
            else:
                flag = True
                data_o[0].append(data_i[0][t])
                data_o[1].append(data_i[1][t])
        length = len(data_o[0])
        for t in reversed(range(length)):
            if (data_o[0][t] < ENGY_THRES * avg_engy):
                data_o[0].pop()
                data_o[1].pop()
            else:
                break
        len1 = len(data_o[0])
        len2 = len(data_o[1])
        data_o[0] = data_o[0][len1 // 100 : len1]
        data_o[1] = data_o[1][len2 // 100 : len2]
        data_ret.append(np.array(data_o))
    # f
    # what if we only use dp instead of NN?
    mean = [-0.00368890111513, 0.00842593986922, 0.00261934419705, -0.0325935548158]
    std = [0.0502888828252, 0.0501864781776, 0.0358012690812, 0.0630625440446]
    for i in range(len(data_ret)):
        dp = [[0] * len(data_ret[i][1])] * 4
        pre = [[-1] * len(data_ret[i][1])] * 4
        mx = 0
        pos = 0
        posT = 0
        for j in range(len(data_ret[i][1])):
            y = np.log(data_ret[i][1][j])
            for t in range(4):
                for k in reversed(range(j)):
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
        gap = int(len(data_i[0]) / LEN_DIM)
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
    for i in range(y.shape[0]):
        for j in range(4):
            if (y[i][j] > 0.9):
                y0[i] = j
    return X, y


# In[4]:

raw_train_data, train_label, train_name = read_file('data/train.txt')
raw_test1_data, test1_label, test1_name = read_file('data/test.txt')
raw_test2_data, test2_label, test2_name = read_file('data/test_new.txt')

train_data = preprocess(raw_train_data)
X_train, y_train = data_process(train_data, train_label)
print (X_train.shape, y_train.shape)


test1_data = preprocess(raw_test1_data)
X_test1, y_test1 = data_process(test1_data, test1_label)
print (X_test1.shape, y_test1.shape)

test2_data = preprocess(raw_test2_data)
X_test2, y_test2 = data_process(test2_data, test2_label)
print (X_test2.shape, y_test2.shape)


# ## model

# In[5]:

HIDDEN_DIM = 4
DEPTH = 0 ## doesn't consider the first and final hidden layer
LEARNING_RATE = 1e-3
REG = 0
Dtype = np.float32
X_DIM = len(X_train[0])
minibatch_size = 20

EPOCH = 3000
SHOW_FREQ = 10
TEST_FREQ = 20


# In[9]:

from cntk.initializer import normal

TcmpS = datetime.datetime.now()

Xs = input_variable((X_DIM), Dtype)
ys = input_variable((4), Dtype)

W1 = parameter(shape=(X_DIM, HIDDEN_DIM), init=np.float32(np.random.normal(0, 0.1, [X_DIM, HIDDEN_DIM])))#normal(0.1))
b1 = parameter(shape=(HIDDEN_DIM),init=0.1)

cur_h = relu(times(Xs, W1) + b1)
#cur_h = times(Xs, W1) + b1

'''for i in range(DEPTH):
    W_i = parameter(shape=(HIDDEN_DIM, HIDDEN_DIM))
    b_i = parameter(shape=(HIDDEN_DIM))
    cur_h = relu(times(cur_h, W_i)+b_i)'''
    
W_out = parameter(shape=(HIDDEN_DIM, 4), init=np.float32(np.random.normal(0, 0.1, [HIDDEN_DIM, 4])))#normal(0.1))
out_num = times(cur_h, W_out)
score = softmax(out_num)

loss = cross_entropy_with_softmax(score, ys)
#eval_error = cross_entropy_with_softmax(score, ys)
eval_error = classification_error(score, ys)

learning_rate = 1e-3
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
#lr_schedule = learning_rate_schedule([(5000*minibatch_size,learning_rate), (1,learning_rate/100)], UnitType.minibatch)


#learner = sgd(score.parameters, lr_schedule)

mom_schedule = momentum_schedule(0.9)
#var_mom_schedule = momentum_schedule(0.999)
learner = adam_sgd(score.parameters, lr_schedule, mom_schedule, l2_regularization_weight=0)
#learner = momentum_sgd(score.parameters, lr_schedule, mom_schedule)

#lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
#learner = adagrad(score.parameters, lr_schedule)

trainer = Trainer(score, loss, eval_error, [learner])


TcmpE = datetime.datetime.now()
# ## Training

# In[10]:

# Run the trainer and perform model training
max_acc2 = 0
acc_all = []
TtrainS = datetime.datetime.now()
for epoch in range(1, EPOCH+1):
    # Specify the input variables mapping in the model to actual minibatch data for training
    for i in range(int(len(X_train) / minibatch_size)):
        #i=0
        trainer.train_minibatch({Xs:X_train[i*minibatch_size:(i+1)*minibatch_size], ys: y_train[i*minibatch_size:(i+1)*minibatch_size]})
    #trainer.train_minibatch({Xs:X_train, ys: y_train})
    
    if (epoch % SHOW_FREQ == 0):
        cur_loss = get_train_loss(trainer)
        acc = get_train_eval_criterion(trainer)
        print("{}/{}, loss = {}, acc = {}".format(epoch, EPOCH, cur_loss, 1-acc))
        #print("{}/{}, loss = {}".format(epoch, EPOCH, trainer.test_minibatch({Xs : X_train[:20], ys : y_train[:20]}) ))
        
    if (epoch % TEST_FREQ == 0):
        print (1-trainer.test_minibatch({Xs : X_test1, ys : y_test1}) )
        acc2 = 1-trainer.test_minibatch({Xs : X_test2, ys : y_test2})
        print ( acc2 )
        acc_all.append(acc2)
        if acc2 > max_acc2:
            max_acc2 = acc2
TtrainE = datetime.datetime.now()

plt.figure(figsize=(15, 9))
plt.plot(np.array(range(len(acc_all)))*20, acc_all, linewidth=1.5)
plt.xlabel('epoch')
plt.ylabel('test acc')
plt.show()

print ("Best acc2 = {}".format(max_acc2))
print ("Compiling time: {}".format(TcmpE - TcmpS))
print ("Training {} epoch time: {}".format(EPOCH, TtrainE - TtrainS))

