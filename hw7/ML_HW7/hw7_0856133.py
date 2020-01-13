import imageio
import numpy as np
from os import listdir
from os.path import isfile, join

imgRowCount = 0
imgColCount = 0
eigenFaceCount = 25

def PCA(img):
    u = np.mean(img, axis=0)
    h = np.ones(imgRowCount)
    B = np.subtract(img, np.reshape(h, (imgRowCount, 1)) @ np.reshape(u, (1, imgColCount)))
    L = (B.T @ B) / (imgColCount - 1)
    lamb , V = np.linalg.eigh(L)
    lamb = np.flip(lamb)
    V = np.flip(V, axis=1)
    U = V @ B.T
    U = U.T
    W = U[:, :eigenFaceCount]
    D = np.zeros((imgColCount, imgColCount))
    np.fill_diagonal(D, lamb)
    #z = B @ W.T @ W
    return u, h, B, L, lamb, V, W


if __name__ == "__main__":
    myPath = './Yale_Face_Database/Training'
    onlyFiles = [join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f))]
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
    uIM, hIM, BIM, LIM, lambIM, VIM, WIM = PCA(im)
    eigenFace = WIM.T
    eigenFace = np.reshape(eigenFace, (25, 231, 195))
    for num in range(len(eigenFace)):
        fileName = './picture/eigenFace[%d].png' % num
        imageio.imwrite(fileName, eigenFace[num])
