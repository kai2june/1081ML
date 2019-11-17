import mlxtend.data
import numpy as np

def loadFile():
    train_image, train_label = mlxtend.data.loadlocal_mnist(
        images_path="./train-images-idx3-ubyte",
        labels_path="./train-labels-idx1-ubyte")
    test_image, test_label = mlxtend.data.loadlocal_mnist(
        images_path="./t10k-images-idx3-ubyte",
        labels_path="./t10k-labels-idx1-ubyte")
    return train_image[:100, ], train_label[:100], test_image[:100, ], test_label[:100]

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

# parameter:
    # lambd[10] : lambd0, lambd1, ..., lambd8, 1-lambd0-lambd1-...-lambd8
    # , w[10][60000]
    # , P[10][28][28]
lambd = np.ones(10)
w = np.ones((10, len(train_image)))
P = np.ones((10, 28, 28))
for i in range(10):
    lambd[i] = 0.1
    print(lambd[i], " ", end='')
print("\n")
#lambd[9] = 1-lambd[0]-lambd[1]-lambd[2]-lambd[3]-lambd[4]-lambd[5]-lambd[6]-lambd[7]-lambd[8]
#print(lambd[9])
for i in range(10):
    for j in range(len(train_image)):
        w[i][j] = 0.1
        # if i == 9:
        #     w[9][j] = 1-w[0][j]-w[1][j]-w[2][j]-w[3][j]-w[4][j]-w[5][j]-w[6][j]-w[7][j]-w[8][j]
        print(w[i][j])

for i in range(10):
    for j in range(28):
        for k in range(28):
            P[i][j][k] = 0.3

print("=============E step===============")
# E-step
for i in range(len(train_image)):
    print("Picture[%d] : \n" % i)
    # for s in range(28):
    #     for t in range(28):
    #         print(train_image[i][s][t], " ", end='')
    #     print("\n")
    denominator = 0
    for category in range(10):
        numerator = lambd[category]
        for j in range(28):
            for k in range(28):
                numerator *= np.power(P[category][j][k], train_image[i][j][k]) * np.power((1-P[category][j][k]), (1-train_image[i][j][k]))
        print(numerator)
        w[category][i] = numerator
        denominator += numerator
    for category in range(10):
        w[category][i] /= denominator
        print(w[category][i])

print("=============M step===============")
print("=============update lambd===========")
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
    print("partial_lambd[%d] : numerator, denominator, lambdMLE[%d]" % (category, category), numerator, denominator, lambdMLE[category])
lambdMLE[9] = 1-lambdMLE[0]-lambdMLE[1]-lambdMLE[2]-lambdMLE[3]-lambdMLE[4]-lambdMLE[5]-lambdMLE[6]-lambdMLE[7]-lambdMLE[8]
print("partial_lambd[9] :", lambdMLE[9])
lambd = lambdMLE

print("==============update P================")
# update P
for category in range(10):
    print("category[%d] : " % category)
    for row in range(28):
        for col in range(28):
            numerator = 0
            denominator = 0
            for i in range(len(train_image)):
                numerator += w[category][i]*train_image[i][row][col]
                denominator += w[category][i]
            #print(numerator, denominator)
            P[category][row][col] = numerator / denominator
            print(P[category][row][col], " ", end='')
        print("\n")

# show result
for category in range(10):
    print("class %d:" % category)
    for i in range(28):
        for j in range(28):
            if P[category][i][j] >= 0.49:
                print(1, end='')
            else:
                print(0, end='')
        print("\n")

# if __name__ == "__main__":
#     EM()
