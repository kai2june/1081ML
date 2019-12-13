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
    elem = []
    for pixel in tmp:
        pixel = float(pixel)
        elem.append(pixel)
    x.append(elem)
#print(x[0])

yt = []
for line in Y_test.readlines():
    line = int(line)
    yt.append(line)

xt = []
for line in X_test.readlines():
    tmp = line.split(',')
    elem = []
    for pixel in tmp:
        pixel = float(pixel)
        elem.append(pixel)
    xt.append(elem)

# # q2_3
# x_train_arr = []
# for line in X_train.readlines():
#     tmp = line.split(',')
#     elem = []
#     for pixel in tmp:
#         pixel = float(pixel)
#         elem.append(pixel)
#     x_train_arr.append(elem)
#
# x_test_arr = []
# for line in X_test.readlines():
#     tmp = line.split(',')
#     elem = []
#     for pixel in tmp:
#         pixel = float(pixel)
#         elem.append(pixel)
#     x_test_arr.append(elem)

def linear_kernel(xa, xb):
    xa = np.array(xa)
    xb = np.array(xb)
    #print("xa {}".format(xa))
    #print(xa.T @ xb)
    return xa.T @ xb

def RBF_kernel(xa, xb):
    sigma = 1
    xa_subtract_xb = [xa[i] - xb[i] for i in range(len(xa))]
    xa_subtract_xb = np.array([a-b for (a, b) in zip(xa, xb)])
    #print("\nxa {}\nxb {}\n xa_subtract_xb {}".format(xa, xb, xa_subtract_xb))
    #print(xa_subtract_xb.T @ xa_subtract_xb/2/sigma/sigma)
    return xa_subtract_xb.T @ xa_subtract_xb/2/sigma/sigma

#linear_kernel([7,8,9], [1,2,4])
#RBF_kernel([7,8,9], [1,2,4])

# K = []
# train_count = len(x)
# for i in range(3):
#     tmp = []
#     tmp.append(i+1)
#     for j in range(3):
#         k_linear = linear_kernel(x[i], x[j])
#         k_RBF = RBF_kernel(x[i], x[j])
#         tmp.append(k_linear + k_RBF)
#     K.append(tmp)
# print(K)
#model = svm_train(y, K, '-t 4 -b 1 -s 0')

beta = 0.2
K = []
train_count = len(x)
for i in range(train_count):
    tmp = []
    tmp.append(i+1)
    for j in range(train_count):
        k_linear = linear_kernel(x[i], x[j])
        k_RBF = RBF_kernel(x[i], x[j])
        tmp.append(k_linear + k_RBF)
    tmp[i+1] += beta
    K.append(tmp)
print(K)

K_test = []
test_count = len(xt)
for i in range(test_count):
    tmp = []
    tmp.append(i+1)
    for j in range(train_count):
        k_linear = linear_kernel(xt[i], x[j])
        k_RBF = RBF_kernel(xt[i], x[j])
        tmp.append(k_linear + k_RBF)
    K_test.append(tmp)
print(K_test)

model = svm_train(y, K, '-t 4 -b 1 -s 0')
p_label, p_acc, p_val = svm_predict(yt, K_test, model)