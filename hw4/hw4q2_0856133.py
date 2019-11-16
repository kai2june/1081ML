import mlxtend.data
import numpy as np

def loadFile():
    train_image, train_label = mlxtend.data.loadlocal_mnist(
        images_path="./train-images-idx3-ubyte",
        labels_path="./train-labels-idx1-ubyte")
    test_image, test_label = mlxtend.data.loadlocal_mnist(
        images_path="./t10k-images-idx3-ubyte",
        labels_path="./t10k-labels-idx1-ubyte")
    return train_image[:20, ], train_label[:20], test_image[:10, ], test_label[:10]

def binning(image):
    binning_image = np.zeros((len(image), len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] >= 0 and image[i][j] < 128:
                binning_image[i][j] = 0
            elif image[i][j] >= 128 and image[i][j] < 256:
                binning_image[i][j] = 1
    return binning_image

def EM():
    # picture
    train_image, train_label, test_image, test_label = loadFile()
    #print("train_image: ", train_image.shape)
    #print("train_label: ", train_label.shape)
    #print("test_image: ", test_image.shape)
    #print("test_label: ", test_label.shape)
    train_image = binning(train_image)
    test_image = binning(test_image)
    #print("train_image[0] = \n", train_image[0])
    #print("binning_image[0] = \n", binning_train_image[0])
    # parameter:
        # lambd[10] : lambd0, lambd1, ..., lambd8, 1-lambd0-lambd1-...-lambd8
        # , w[10][60000]
        # , P[10][28][28]
    lambd = np.ones(10)
    w = np.ones((10, len(train_image)))
    P = np.ones((10, 28, 28))
    for i in range(10):
        lambd[i] = 0.1
    for i in range(10):
        for j in range(len(train_image)):
            w[i][j] = 0.1
    for i in range(10):
        for j in range(28):
            for k in range(len(28)):
                P[i][j][k] = 0.5
    # E-step
    train_image = np.reshape(train_image, (-1, 28, 28))
    test_image = np.reshape(test_image, (-1, 28, 28))
    for i in range(len(train_image)):
        denominator = 0
        for category in range(10):
            if category == 9:
                tmp = 1-lambd[0]-lambd[1]-lambd[2]-lambd[3]-lambd[4]-lambd[5]-lambd[6]-lambd[7]-lambd[8]
            else:
                tmp = lambd[category]
            for j in range(28):
                for k in range(28):
                    tmp *= np.power(P[category][j][k], train_image[i][j][k])
            w[category][i] = tmp
            denominator += tmp
        for category in range(10):
            w[category][i] /= denominator

    # M-step
    lambdMLE = np.ones(10)
    for category in range(9):
        for s in range(9):
            if s == category:
                continue
            else:
                lambdMLE[category] -= lambd[s]

        numerator = 0
        denominator = 0

        for i in range(len(train_image)):
            numerator += w[category][i]
            tmp = 1
            for s in range(9):
                if s == category:
                    continue
                else:
                    tmp -= w[s][i]
            denominator += tmp
        lambdMLE[category] *= numerator/denominator
    lambdMLE[9] = 1-lambdMLE[0]-lambdMLE[1]-lambdMLE[2]-lambdMLE[3]-lambdMLE[4]-lambdMLE[5]-lambdMLE[6]-lambdMLE[7]-lambdMLE[8]
    lambd = lambdMLE

    # update P
    for category in range(10):
        for row in range(28):
            for col in range(28):
                numerator = 0
                denominator = 0
                for i in range(len(train_image)):
                    numerator += w[category][i]*train_image[i][row][col]
                    denominator += w[category][i]
                P[category][row][col] = numerator / denominator



if __name__ == "__main__":
    EM()
