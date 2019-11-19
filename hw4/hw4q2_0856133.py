import mlxtend.data
import numpy as np

def loadFile():
    train_image, train_label = mlxtend.data.loadlocal_mnist(
        images_path="./train-images-idx3-ubyte",
        labels_path="./train-labels-idx1-ubyte")
    test_image, test_label = mlxtend.data.loadlocal_mnist(
        images_path="./t10k-images-idx3-ubyte",
        labels_path="./t10k-labels-idx1-ubyte")
    return train_image[:30, ], train_label[:30], test_image[:100, ], test_label[:100]

def binning(image):
    binning_image = np.zeros((len(image), 28, 28))
    for i in range(len(image)):
        for j in range(28):
            for k in range(28):
                if image[i][j][k] >= 0 and image[i][j][k] < 128:
                    binning_image[i][j][k] = 0
                elif image[i][j][k] >= 128 and image[i][j][k] < 256:
                    binning_image[i][j][k] = 1
    return binning_image

# def EM():
#     # picture

train_image, train_label, test_image, test_label = loadFile()
train_image = np.reshape(train_image, (-1, 28, 28))
test_image = np.reshape(test_image, (-1, 28, 28))
# for i in range(28):
#     for j in range(28):
#         print(train_image[0][i][j], " ", end='')
#     print("\n")
train_image = binning(train_image)
test_image = binning(test_image)
# for i in range(28):
#     for j in range(28):
#         print(train_image[0][i][j], " ", end='')
#     print("\n")
full_train_image = train_image
#print(full_train_image)
# for i in range(len(full_train_image)):
#     for j in range(28):
#         for k in range(28):
#             print(full_train_image[i][j][k], " ", end='')
#         print("\n")

# parameter:
    # w[10][60000]
    # , lambd[10] : lambd0, lambd1, ..., lambd8, 1-lambd0-lambd1-...-lambd8
    # , P[10][28][28]
w = np.ones((10, len(train_image)))
lambd = np.ones(10)
P = np.ones((10, 28, 28))

# for i in range(10):
#     for j in range(len(train_image)):
#         w[i][j] = 0.1
#         # if i == 9:
#         #     w[9][j] = 1-w[0][j]-w[1][j]-w[2][j]-w[3][j]-w[4][j]-w[5][j]-w[6][j]-w[7][j]-w[8][j]
#         print(w[i][j])
for i in range(len(train_image)):
    w[0][i] = 5923/60000
    w[1][i] = 6742/60000
    w[2][i] = 5958/60000
    w[3][i] = 6131/60000
    w[4][i] = 5842/60000
    w[5][i] = 5421/60000
    w[6][i] = 5918/60000
    w[7][i] = 6265/60000
    w[8][i] = 5851/60000
    w[9][i] = 5949/60000
#print(w)

# for i in range(10):
#     lambd[i] = 0.1
#     print(lambd[i], " ", end='')
# print("\n")
# lambd[9] = 1-lambd[0]-lambd[1]-lambd[2]-lambd[3]-lambd[4]-lambd[5]-lambd[6]-lambd[7]-lambd[8]
# print(lambd[9])
lambd[0] = 5923
lambd[1] = 6742
lambd[2] = 5958
lambd[3] = 6131
lambd[4] = 5842
lambd[5] = 5421
lambd[6] = 5918
lambd[7] = 6265
lambd[8] = 5851
lambd[9] = 5949
for i in range(10):
    lambd[i] /= 60000
    #print("lambd[%d] :" % i, lambd[i])


for i in range(10):
    for j in range(28):
        for k in range(28):
            P[i][j][k] = 0.05

converge_iteration = 0
for epoch in range(20):
    converge_iteration += 1
    print("===============================epoch[%d]==================================" % epoch)
    #train_image = full_train_image[300*(epoch):300*(epoch+1), :, :]
    # for i in range(len(train_image)):
    #     for j in range(28):
    #         for k in range(28):
    #             print(train_image[i][j][k], " ", end='')
    #         print("\n")
    #     print("\n")
    #print("=============E step===============")
    # E-step
    for i in range(len(train_image)):
        #print("Picture[%d] : \n" % i)
        # for s in range(28):
        #     for t in range(28):
        #         print(train_image[i][s][t], " ", end='')
        #     print("\n")
        denominator = 0
        for category in range(10):
            numerator = lambd[category]
            for j in range(28):
                for k in range(28):
                    numerator *= np.maximum(np.power(P[category][j][k], train_image[i][j][k]), 0.05) \
                                 * np.maximum(np.power((1-P[category][j][k]), (1-train_image[i][j][k])), 0.05)
                    #print("numerator:", numerator)
            w[category][i] = numerator
            denominator += numerator
            #print("denominator", denominator)
        for category in range(10):
            w[category][i] /= denominator
            #print(w[category][i])
    #print(w)
    #print("=============M step===============")
    #print("=============update lambd===========")
    # M-step
    # update lambd
    lambdMLE = np.zeros(10)
    for category in range(9):
        lambdMLE[category] = lambd[category] + lambd[9]
    #     print(lambdMLE[category])
    # print(lambdMLE[9])
        numerator = 0
        denominator = 0
        for i in range(len(train_image)):
        #     print("Picture[%d] : \n" % i)
        #     for s in range(28):
        #         for t in range(28):
        #             print(train_image[i][s][t], " ", end='')
        #         print("\n")
            numerator += w[category][i]
            denominator += w[category][i] + w[9][i]
        lambdMLE[category] *= numerator/denominator
        #print("partial_lambd[%d] : numerator, denominator, lambdMLE[%d]" % (category, category), numerator, denominator, lambdMLE[category])
    lambdMLE[9] = 1-lambdMLE[0]-lambdMLE[1]-lambdMLE[2]-lambdMLE[3]-lambdMLE[4]-lambdMLE[5]-lambdMLE[6]-lambdMLE[7]-lambdMLE[8]
    #print("partial_lambd[9] :", lambdMLE[9])
    lambd = lambdMLE

    # for i in range(10):
    #     for j in range(28):
    #         for k in range(28):
    #             print(P[i][j][k], " ", end='')
    #         print("\n")
    #print("==============update P================")
    # update P
    for category in range(10):
        #print("category[%d] : " % category)
        for row in range(28):
            for col in range(28):
                numerator = 0
                denominator = 0
                for i in range(len(train_image)):
                    #print(w[category][i], train_image[i][row][col])
                    numerator += w[category][i]*train_image[i][row][col]
                    denominator += w[category][i]
                #print(numerator, denominator)
                P[category][row][col] = numerator / denominator
                #print(P[category][row][col], " ", end='')
            #print("\n")
#     for i in range(10):
#         for j in range(28):
#             for k in range(28):
#                 print(P[i][j][k], " ", end='')
#             print("\n")

    if epoch == 0:
        tmp = np.copy(P)
    difference = 0
    print("==================imagination picture======================")
    # show result
    for category in range(10):
        print("class %d:" % category)
        for i in range(28):
            for j in range(28):
                if P[category][i][j] >= 0.5:
                    print(1, end='')
                    if tmp[category][i][j] < 0.5:
                        difference += 1
                else:
                    print(0, end='')
                    if tmp[category][i][j] >= 0.5:
                        difference += 1
            print("\n")
    tmp = np.copy(P)
    print("No. of Iteration: %d, Difference: %f" % (epoch, difference) )
    if difference <= 10 and epoch >=15 :
        break

rlt = np.array([3, 2, 8, 4, 9, 5, 7, 1, 6, 0])
order = np.array([9, 7, 1, 0, 3, 5, 8, 6, 2, 4])
for i in range(10):
    print("labeled class %d:" % i)
    for j in range(28):
        for k in range(28):
            if P[order[i]][j][k] >= 0.5:
                print(1, end='')
            else:
                print(0, end='')
        print("\n")
################confusion matrix###############
count = np.zeros(10)
for i in range(len(train_label)):
    count[train_label[i]] += 1

true_positive = 0
for i in range(10):
    TP, FN, FP, TN = 0, 0, 0, 0
    for j in range(len(train_image)):
        if train_label[j] == i and w[i][j] >= 0.5:
            TP += 1
            true_positive += 1
        elif train_label[j] == i and w[i][j] < 0.5:
            FN += 1
        elif train_label[j] != i and w[i][j] >= 0.5:
            FP += 1
        elif train_label[j] != i and w[i][j] < 0.5:
            TN += 1
    print("Confusion Matrix %d:" % i)
    print("                             Predict number %d, Predict number %d" %(i, i))
    print("Ground truth number %d:          %d                  %d" % (i, TP, FN))
    print("Ground truth number not %d:      %d                  %d" % (i, FP, TN))
    print("Sensitivity (Successfully predict number %d): %f" % (i, TP/(TP + FN)))
    print("Specificity (Successfully predict not number %d): %f" % (i, TN/(TN + FP)))
    print("==============================================================================")
print("Total iteration to converge: %d" % converge_iteration)
print("Total error rate: %f" % (1 - true_positive/len(train_image)))
# if __name__ == "__main__":
#     EM()
