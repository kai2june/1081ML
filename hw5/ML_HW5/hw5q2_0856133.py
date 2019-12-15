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

# # q2_2 linear kernel (should be without gamma, anyway, this result is the same as without gamma)
# C = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
# G = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
# # C = [2, 8]
# # G = [0.5]
# max_acc = 0.0
# max_c = 0.0
# max_gamma = 0.0
# for c in C:
#     for gamma in G:
#         model = svm_train(y, x, '-t 0 -b 1 -s 0 -c {} -g {} -v 5'.format(c, gamma))
#         print('c = {}, gamma = {}, accuracy = {}'.format(c, gamma, model))
#         if model > max_acc:
#             max_acc = model
#             max_c = c
#             max_gamma = gamma
# print("max_acc", max_acc, "max_c", max_c, "max_gamma", max_gamma)
# model = svm_train(y, x, '-t 0 -b 1 -s 0 -c {} -g {}'.format(max_c, max_gamma))
# p_label, p_acc, p_val = svm_predict(yt, xt, model)

# q2_2 linear kernel
C = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
#C = [2, 8]
max_acc = 0.0
max_c = 0.0
for c in C:
    model = svm_train(y, x, '-t 0 -b 1 -s 0 -c {} -v 5'.format(c))
    print('c = {}, accuracy = {}'.format(c, model))
    if model > max_acc:
        max_acc = model
        max_c = c
print("max_acc", max_acc, "max_c", max_c)
model = svm_train(y, x, '-t 0 -b 1 -s 0 -c {}'.format(max_c))
p_label, p_acc, p_val = svm_predict(yt, xt, model)

# q2_2 RBF kernel
C = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
G = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
# C = [2, 8]
# G = [0.5]
max_acc = 0.0
max_c = 0.0
max_gamma = 0.0
for c in C:
    for gamma in G:
        model = svm_train(y, x, '-t 2 -b 1 -s 0 -c {} -g {} -v 5'.format(c, gamma))
        print('c = {}, gamma = {}, accuracy = {}'.format(c, gamma, model))
        if model > max_acc:
            max_acc = model
            max_c = c
            max_gamma = gamma
print("max_acc", max_acc, "max_c", max_c, "max_gamma", max_gamma)
model = svm_train(y, x, '-t 2 -b 1 -s 0 -c {} -g {}'.format(max_c, max_gamma))
p_label, p_acc, p_val = svm_predict(yt, xt, model)