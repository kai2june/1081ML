import numpy as np
from libsvm.svmutil import *
from libsvm.svm import *

Y_train = open('Y_train.csv', 'r')
X_train = open('X_train.csv', 'r')
Y_test = open('Y_test.csv', 'r')
X_test = open('X_test.csv', 'r')

y = []
for line in Y_train.readlines():
    line = int(line)
    y.append(line)
#print(y)

x = []
for line in X_train.readlines():
    tmp = line.split(',')
    elem = dict()
    i = 1
    for pixel in tmp:
        pixel = float(pixel)
        elem.update({i: pixel})
        i += 1
    x.append(elem)
#print(x)

yt = []
for line in Y_test.readlines():
    line = int(line)
    yt.append(line)

xt = []
for line in X_test.readlines():
    tmp = line.split(',')
    elem = dict()
    i = 1
    for pixel in tmp:
        pixel = float(pixel)
        elem.update({i: pixel})
        i += 1
    xt.append(elem)

# linear kernel
#prob = svm_problem(y, x)
#param = svm_parameter('-t 0 -b 1')
model = svm_train(y, x, '-t 0 -b 1')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
#print(p_label)

# polynomial kernel
model = svm_train(y, x, '-t 1 -b 1')
p_label, p_acc, p_val = svm_predict(yt, xt, model)

# RBF kernel
model = svm_train(y, x, '-t 2 -b 1')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
