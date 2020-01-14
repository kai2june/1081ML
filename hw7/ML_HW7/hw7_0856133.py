import imageio
import numpy as np
from os import listdir
from os.path import isfile, join

imgRowCount = 0
imgColCount = 0
eigenFaceCount = 25

def PCA(img, imgTest):
    # training eigenface
    # u = np.mean(img, axis=0)
    # h = np.ones(imgRowCount)
    # B = np.subtract(img, np.reshape(h, (imgRowCount, 1)) @ np.reshape(u, (1, imgColCount)))
    u = np.mean(img, axis=1)  # 45045
    h = np.ones(imgColCount)  # 135
    B = np.subtract(img, np.reshape(u, (imgRowCount, 1)) @ np.reshape(h, (1, imgColCount)))  # 45045*135
    L = (B.T @ B) / (imgColCount - 1)  # 135*135
    lamb , V = np.linalg.eigh(L)
    lamb = np.flip(lamb)
    V = np.flip(V, axis=1)
    U = V @ B.T  # 135*45045
    U = U.T  # 45045*135
    W = U[:, :eigenFaceCount]  # 45045*25
    D = np.zeros((imgColCount, imgColCount))
    np.fill_diagonal(D, lamb)

    # training reconstruction
    omega = W.T @ B # 25*135
    reconstruction = W @ omega  # 45045*135
    reconstruction = reconstruction.T
    reconstruction = reconstruction / 25

    # testing eigenface

    return u, h, B, L, lamb, V, W, omega, reconstruction

if __name__ == "__main__":

    # Training image
    myPath = './Yale_Face_Database/Training'
    onlyFiles = [join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f))]
    onlyFiles.sort()
    im = np.array([])
    for num in range(len(onlyFiles)):
        pic1D = imageio.imread(onlyFiles[num])
        pic1D = np.asarray(pic1D)
        pic1D = np.ravel(pic1D)
        im = np.append(im, pic1D)
    im = np.asarray(im)
    im = im.reshape((135, 45045))
    im = im.T
    imgRowCount = len(im)
    imgColCount = len(im[0])

    # Test image
    testPath = './Yale_Face_Database/Testing'
    testFiles = [join(testPath, f) for f in listdir(testPath) if isfile(join(testPath, f))]
    testFiles.sort()
    imTest = np.array([])
    for num in range(len(testFiles)):
        pic1D = imageio.imread(testFiles[num])
        pic1D = np.asarray(pic1D)
        pic1D = np.ravel(pic1D)
        imTest = np.append(imTest, pic1D)
    imTest = np.asarray(imTest)
    imTest = np.reshape(imTest, (30, 45045))
    imTest = imTest.T

    # PCA
    uIM, hIM, BIM, LIM, lambIM, VIM, WIM, omegaIM, reconstructionIM = PCA(im, imTest)
    eigenFace = WIM.T
    eigenFace = np.reshape(eigenFace, (25, 231, 195))
    for num in range(len(eigenFace)):
        fileName = './picture/PCA/eigenFace[%d].png' % num
        imageio.imwrite(fileName, eigenFace[num])

    for num in range(len(reconstructionIM)):
        tmp = np.reshape(reconstructionIM[num], (231, 195))
        fileName = './picture/PCA/reconstruction[%d].png' % num
        imageio.imwrite(fileName, tmp)
