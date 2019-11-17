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

def EM():
    # picture
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
    for i in range(10-1):
        lambd[i] = 0.02*i + 0.01
        #print(lambd[i], " ", end='')
    lambd[9] = 1-lambd[0]-lambd[1]-lambd[2]-lambd[3]-lambd[4]-lambd[5]-lambd[6]-lambd[7]-lambd[8]
    #print(lambd[9])
    for i in range(10):
        for j in range(len(train_image)):
            w[i][j] = 0.02*(9-i) + 0.01
            w[9][j] = 1-w[0][j]-w[1][j]-w[2][j]-w[3][j]-w[4][j]-w[5][j]-w[6][j]-w[7][j]-w[8][j]
            #print(w[i][j])

    for i in range(10):
        for j in range(28):
            for k in range(28):
                P[i][j][k] = 0.3

    for epoch in range(10):
        train_image = full_train_image[10*epoch: 10*(epoch+1), :, :]
        # E-step
        for i in range(len(train_image)):
            print("Picture[%d] : \n" % i)
            for s in range(28):
                for t in range(28):
                    print(train_image[i][s][t], " ", end='')
                print("\n")
            denominator = 0
            for category in range(10):
                if category == 9:
                    tmp = 1-lambd[0]-lambd[1]-lambd[2]-lambd[3]-lambd[4]-lambd[5]-lambd[6]-lambd[7]-lambd[8]
                else:
                    tmp = lambd[category]
                for j in range(28):
                    for k in range(28):
                        tmp *= np.power(P[category][j][k], train_image[i][j][k])*np.power((1-P[category][j][k]), train_image[i][j][k])
                print(tmp)
                w[category][i] = tmp
                denominator += tmp
            for category in range(10):
                w[category][i] /= denominator
                print(w[category][i])

        # M-step
        # update lambd
        lambdMLE = np.ones(10)
        for category in range(9):
            for s in range(9):
                if s == category:
                    continue
                else:
                    lambdMLE[category] -= lambd[s]
            #print(lambdMLE[category])
            numerator = 0
            denominator = 0

            for i in range(len(train_image)):
                print("Picture[%d] : \n" % i)
                for s in range(28):
                    for t in range(28):
                        print(train_image[i][s][t], " ", end='')
                    print("\n")
                numerator += w[category][i]
                tmp = 1
                for s in range(9):
                    if s == category:
                        continue
                    else:
                        tmp -= w[s][i]
                print(tmp)
                denominator += tmp
            print("numerator, denominator", numerator, denominator)
            lambdMLE[category] *= numerator/denominator
        lambdMLE[9] = 1-lambdMLE[0]-lambdMLE[1]-lambdMLE[2]-lambdMLE[3]-lambdMLE[4]-lambdMLE[5]-lambdMLE[6]-lambdMLE[7]-lambdMLE[8]
        print("lambdMLE :", lambdMLE)
        lambd = lambdMLE
        print("lambd :", lambd)

        # update P
        for category in range(10):
            for row in range(28):
                for col in range(28):
                    numerator = 0
                    denominator = 0
                    for i in range(len(train_image)):
                        numerator += w[category][i]*train_image[i][row][col]
                        denominator += w[category][i]
                    #print(numerator, denominator)
                    P[category][row][col] = numerator / denominator
                #print(P[category][row])

        # # show result
        # print("===================Round %d : =======================" % epoch)
        # for category in range(10):
        #     print("class %d:" % category)
        #     for i in range(28):
        #         for j in range(28):
        #             if P[category][i][j] >= 0.5:
        #                 print(1, end='')
        #             else:
        #                 print(0, end='')
        #         print("\n")

    # for i in range(10):
    #     for j in range(len(train_image)):
    #         print("w[%d][%d]: " % (i, j), w[i][j])

    # print("lambd: ", lambd)


if __name__ == "__main__":
    EM()
