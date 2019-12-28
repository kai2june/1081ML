import imageio
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

# def user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe, gamma_s=1/100000, gamma_c=1/100000):
#     sx = np.asarray(sx)
#     sx_apostrophe = np.asarray(sx_apostrophe)
#     S = np.array([a-b for (a, b) in zip(sx, sx_apostrophe)])
#     S = np.array(np.sum([np.power(a, 2) for a in S]))
#     cx = np.asarray(cx)
#     cx_apostrophe = np.asarray(cx_apostrophe)
#     #print("cx", cx)
#     #print("cx_apostrophe", cx_apostrophe)
#     C = np.array([a-b for (a, b) in zip(cx, cx_apostrophe)])
#     C = np.array(np.sum([np.power(a, 2) for a in C]))
#     #print("S: ", S)
#     #print("C: ", C)
#     return np.exp(-gamma_s*S) * np.exp(-gamma_c*C)
#
# def distance_on_kernel(img, cluster, cluster_count):
#     distanceOnKernel = np.zeros((100,100, len(cluster_count)))
#     gramMatrix = np.zeros(len(cluster_count))
#     newCluster = np.zeros((100, 100))
#     W = np.zeros((100, 100, 100, 100))
#     D = np.zeros((100, 100))
#     for i in range(100):
#         for j in range(100):
#             newCluster[i][j] = -1
#
#     # apply Wei-chen Chiu's Unsupervised_Learning.pdf p.22 formula
#     print("Computing formula...")
#     for i in range(100):
#         print("Computing row %d:" %i)
#         for j in range(100):
#             # left term
#             sx = [i, j]
#             cx = [img[i][j][0], img[i][j][1], img[i][j][2]]
#             kxx = user_defined_kernel(sx, sx, cx, cx)
#             for p in range(len(cluster_count)):
#                 distanceOnKernel[i][j][p] = kxx
#
#             # middle term, gram matrix
#             middleTerm = np.zeros(3)
#             for s in range(100):
#                 for t in range(100):
#                     sx_apostrophe = [s, t]
#                     cx_apostrophe = [img[s][t][0], img[s][t][1], img[s][t][2]]
#                     #print("cluster[s][t]", int(cluster[s][t]))
#                     tmp = user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe)
#                     ## compute similary matrix W, diagonal matrix D
#                     if i == s and j == t:
#                         pass
#                     else:
#                         W[i][j][s][t] = tmp
#                         D[i][j] += tmp
#                         #L = ??????????????????????????????????
#                     ###############################################
#                     #print("middleTerm, tmp:", middleTerm[int(cluster[s][t])], tmp)
#                     middleTerm[int(cluster[s][t])] += tmp
#                     if cluster[i][j] == cluster[s][t]:
#                         gramMatrix[int(cluster[i][j])] += tmp
#
#             for c in range(len(cluster_count)):
#                 middleTerm[c] = -2 * middleTerm[c] / cluster_count[int(cluster[i][j])]
#                 distanceOnKernel[i][j][c] += middleTerm[c]
#
#     # right term
#     rightTerm = np.copy(gramMatrix)
#     for i in range(len(rightTerm)):
#         rightTerm[i] = rightTerm[i] / cluster_count[i] / cluster_count[i]
#     for i in range(100):
#         for j in range(100):
#             mini = np.Inf
#             for c in range(len(cluster_count)):
#                 distanceOnKernel[i][j][c] += rightTerm[c]
#                 if distanceOnKernel[i][j][c] < mini:
#                     mini = distanceOnKernel[i][j][c]
#                     newCluster[i][j] = c
#     return newCluster, W, D
#
# def kernel_kmeans(img, k):
#     # initialize centroid
#     centroid = []
#     cluster = np.zeros((100, 100))
#     for i in range(100):
#         for j in range(100):
#             cluster[i][j] = -1
#     i = 0
#     while i < k:
#         tmp = np.random.randint(100, size=2)
#         for j in range(i):
#             while tmp[0] == centroid[j][0] and tmp[1] == centroid[j][1]:
#                 tmp = np.random.randint(100, size=2)
#         centroid.append(tmp)
#         cluster[tmp[0]][tmp[1]] = i
#         i += 1
#     centroid = np.array(centroid)
#     print("Initial centorid: ", centroid)
#
#     # kernel, compute distance between 10000 pixel and 3 centroid
#     K = np.zeros([100, 100])
#     for i in range(100):
#         for j in range(100):
#             sx = [i, j]
#             cx = [img[i][j][0], img[i][j][1], img[i][j][2]]
#             K[i][j] = 0
#             for cen in range(k):
#                 sx_apostrophe = centroid[cen]
#                 cx_apostrophe = [img[centroid[cen][0]][centroid[cen][1]][0], \
#                                  img[centroid[cen][0]][centroid[cen][1]][1], \
#                                  img[centroid[cen][0]][centroid[cen][1]][2]]
#                 tmp = user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe)
#                 if tmp > K[i][j]:
#                     K[i][j] = tmp
#                     cluster[i][j] = cluster[centroid[cen][0]][centroid[cen][1]]
#                 #print(K[i][j])
#
#     # count how many points in each cluster
#     cluster_count = np.zeros(k)
#     for i in range(100):
#         for j in range(100):
#             for p in range(k):
#                 if p == cluster[i][j]:
#                     cluster_count[p] += 1
#                     break
#     print("cluster_count: ", cluster_count)
#
#     # recompute centroid
#     newCluster, W, D = distance_on_kernel(img, cluster, cluster_count)
#     for i in range(100):
#         print("newCluster[%d]" %i)
#         for j in range(100):
#             print(newCluster[i][j])
#
#     # initialize visualization
#     X0 = []
#     Y0 = []
#     X1 = []
#     Y1 = []
#     X2 = []
#     Y2 = []
#     for i in range(100):
#         for j in range(100):
#             if cluster[i][j] == 0:
#                 X0.append(99-i)
#                 Y0.append(j)
#             elif cluster[i][j] == 1:
#                 X1.append(99-i)
#                 Y1.append(j)
#             elif cluster[i][j] == 2:
#                 X2.append(99-i)
#                 Y2.append(j)
#     plt.plot(X0, Y0, 'r.')
#     plt.plot(X1, Y1, 'g.')
#     plt.plot(X2, Y2, 'b.')
#
#     # 1st iteration visualization
#     X0 = []
#     Y0 = []
#     X1 = []
#     Y1 = []
#     X2 = []
#     Y2 = []
#     for i in range(100):
#         for j in range(100):
#             if newCluster[i][j] == 0:
#                 X0.append(99-i)
#                 Y0.append(j)
#             elif newCluster[i][j] == 1:
#                 X1.append(99-i)
#                 Y1.append(j)
#             elif newCluster[i][j] == 2:
#                 X2.append(99-i)
#                 Y2.append(j)
#     plt.plot(X0, Y0, 'r.')
#     plt.plot(X1, Y1, 'g.')
#     plt.plot(X2, Y2, 'b.')
#     plt.show()
#     plt.show()
#     return W, D

def compute_kernel(sub, gammaC=1/100000, gammaS=1/100000):
    #print("sub: ", sub)
    C = sub[:, :3]
    S = sub[:, 3:]
    #print("C: ", C)
    #print("C**2: ", C**2)
    #print("np.sum(C**2, axis=1): ", np.sum(C**2, axis=1))
    #print("S: ", S)
    return np.exp(-gammaC*np.sum(C**2, axis=1)) * np.exp(-gammaS*np.sum(S**2, axis=1))

def compute_gram_matrix(img):
    print("computing gram matrix...")
    #print("img.shape: ", img.shape)
    imgCopy = np.copy(img)
    gramMatrix = np.zeros((10000, 10000))
    for s in range(10000):
        #print("s: ", s)
        sub = np.subtract(img, imgCopy)
        offset = np.arange(10000-s)
        tmp = compute_kernel(sub)
        tmp = tmp[:10000-s]
        gramMatrix[offset, offset+s] = tmp
        #print("gramMatrix:", gramMatrix)
        imgCopy = np.roll(imgCopy, -1, axis=0)
        #print("imgCopy: ", imgCopy)
    gramMatrix = np.maximum(gramMatrix, gramMatrix.T)
    #print("In computer_gram_matrix, gramMatrix= ", gramMatrix)
    return gramMatrix

def compute_kernel_kmeans(gramMatrix, k=2, init=1):
    print("computing kernel kmeans...")
    ## initialize centroid
    #print(gramMatrix.shape)
    centroid2D = []
    cluster = np.zeros((100, 100))
    allCluster = []
    for i in range(100):
        for j in range(100):
            cluster[i][j] = -1

    ### pick centroid, original vs. kmeans++
    if init == 0:
        i = 0
        while i < k:
            tmp = np.random.randint(100, size=2)
            for j in range(i):
                while tmp[0] == centroid2D[j][0] and tmp[1] == centroid2D[j][1]:
                    tmp = np.random.randint(100, size=2)
            centroid2D.append(tmp)
            cluster[tmp[0]][tmp[1]] = i
            i += 1
    elif init == 1:
        i = 0
        firstCentroid = 0
        while i < k:
            if i == 0:
                tmp = np.random.randint(100, size=2)
                centroid2D.append(tmp)
                cluster[tmp[0]][tmp[1]] = i
                firstCentroid = tmp[0]*100 + tmp[1]
            else:
                D = np.copy(gramMatrix[:, firstCentroid])
                D = D**2
                total = np.sum(D)
                P = D / total
                cumulativeP = np.cumsum(P)
                roulette = np.random.rand()
                newCentroid = np.searchsorted(cumulativeP, roulette)
                centroid2D.append(np.array([(newCentroid/100).astype(int), newCentroid % 100]))
            i += 1
    ##########################################
    centroid2D = np.array(centroid2D)
    print("Initial centroid: ", centroid2D)
    centroid1D = np.array([])
    #print("centroid1D: ", centroid1D)
    for [i, j] in centroid2D:
        centroid1D = np.append(centroid1D, i*100+j)
    #print("centroid1D: ", centroid1D)
    centroid1D = centroid1D.astype(int)
    print("centroid1D: ", centroid1D)

    ## initialize cluster label
    cluster = np.ravel(cluster)
    for i in range(10000):
        #print('\n')
        cen = np.array([])
        for j in range(k):
            cen = np.append(cen, gramMatrix[i][centroid1D[j]])
            #print("gramMatrix[%d][%d]: " % (i, centroid1D[j]), gramMatrix[i][centroid1D[j]])
        #print("np.max(cen): ", np.max(cen))
        #print("np.where(np.max(cen)): ", np.where(np.max(cen)))
        cluster[i] = np.where(cen == np.max(cen))[0][0]
        #print("cen[0][0]: ", cluster[i])
    cluster = cluster.astype(int)
    print("Initial:")
    for i in range(k):
        print("clusterCount[%d]: " % i, np.sum(cluster == i))
    allCluster.append(cluster)

    ## iterate until converge, no need to compute left term(=1 for all pixels)
    ## (Prof. Chiu, unsupervised.pdf p.22)
    newCluster = np.zeros(10000)
    #newCluster[0] = -1  ###### in case newCluster equals cluster in 1st iteration
    firstIteration = True
    itr = 0
    while np.any(newCluster != cluster):
        itr += 1
        print("itr %d:" % itr)
        if firstIteration:
            firstIteration = False
        else:
            cluster = np.copy(newCluster)
            allCluster.append(cluster)
        clusterCount = np.array([])
        for i in range(k):
            clusterCount = np.append(clusterCount, np.sum(cluster == i))
        clusterCount = clusterCount.astype(int)
        print("clusterCount[%d] :" % (itr-1), clusterCount)

        ### compute right term
        rightTerm = np.zeros(k)
        pixelToCluster = np.zeros(10000)
        for i in range(len(pixelToCluster)):
            pixelToCluster[i] = np.Inf
        for c in range(k):
            tmp = (cluster == c)
            tmpRight = np.copy(gramMatrix[tmp == 1, :])
            tmpRight = tmpRight[:, tmp == 1]
            tmpRight = np.sum(tmpRight) / clusterCount[c] / clusterCount[c]
            rightTerm[c] = tmpRight

            ### compute middle term
            middleTerm = 0
            for i in range(10000):
                #print("gramMatrix[%d]: " % i, gramMatrix[i])
                tmpMiddle = gramMatrix[i] @ tmp
                #print("tmpMiddle[%d]: " % i, tmpMiddle)
                tmpMiddle = -tmpMiddle * 2 / clusterCount[c]
                middleTerm = tmpMiddle
                if 1 + middleTerm + rightTerm[c] < pixelToCluster[i]:
                    pixelToCluster[i] = 1 + middleTerm + rightTerm[c]
                    #print("pixelToCluster[%d]: " % i, pixelToCluster[i])
                    newCluster[i] = c
        # allCluster.append(newCluster)

    allCluster = np.asarray(allCluster)
    scatter_plot(allCluster)
    return allCluster

def scatter_plot(allCluster):
    print("plotting...")
    allCluster = np.reshape(allCluster, (-1, 100, 100))  ###### Alert: call by value? call by address?
    colorMap = ['r.', 'g.', 'b.', 'y.', 'c.', 'm.', 'k.', 'w.']

    for itr in range(len(allCluster)):
        for c in range(len(np.unique(allCluster[0]))):
            x = np.array([])
            y = np.array([])
            for i in range(99, -1, -1):
                for j in range(100):
                    if allCluster[itr][i][j] == c:
                        x = np.append(x, j)
                        y = np.append(y, i)
            plt.title("kernel kmeans, Iter %d" % itr)
            plt.plot(x, y, colorMap[c])
            figName = "./picture/k%d++/kernelKMeans++Itr%dk%d" % (len(np.unique(allCluster[0])), itr, len(np.unique(allCluster[0])))
            #plt.savefig(figName)
        plt.show()

if __name__ == "__main__":
    #similarityMatrix, diagonalMatrix = kernel_kmeans(im, 3)
    #user_defined_kernel([2,5], [9,9], [2,3,9], [9,2,4])
    #spectral_clustering(im, 3, similarityMatrix, diagonalMatrix)

    im = imageio.imread('./image1.png')
    im = np.array(im)
    im = np.reshape(im, (10000, 3))

    ind = []
    for row in range(100):
        for col in range(100):
            ind.append([row, col])
    ind = np.array(ind)
    im = np.append(im, ind, axis=1)
    gramMatrixInMain = compute_gram_matrix(im)
    print("In main, gramMatrixInMain= ", gramMatrixInMain)

    ## kernel kmeans
    allC = compute_kernel_kmeans(gramMatrixInMain)
    print("kernel kmeans done!!")
    ## spectral clustering


    # pl = np.zeros(10000)
    # for s in range(3000, 3500):
    #     pl[s] = 1
    # for s in range(4000, 4500):
    #     pl[s] = 2
    # for s in range(5000, 5500):
    #     pl[s] = 3
    # for s in range(6000, 6500):
    #     pl[s] = 4
    # for s in range(7000, 7500):
    #     pl[s] = 5
    # scatter_plot(pl)
