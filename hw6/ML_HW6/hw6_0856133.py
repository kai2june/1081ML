import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe, gamma_s=1/100000, gamma_c=1/100000):
    sx = np.asarray(sx)
    sx_apostrophe = np.asarray(sx_apostrophe)
    S = np.array([a-b for (a, b) in zip(sx, sx_apostrophe)])
    S = np.array(np.sum([np.power(a, 2) for a in S]))
    cx = np.asarray(cx)
    cx_apostrophe = np.asarray(cx_apostrophe)
    #print("cx", cx)
    #print("cx_apostrophe", cx_apostrophe)
    C = np.array([a-b for (a, b) in zip(cx, cx_apostrophe)])
    C = np.array(np.sum([np.power(a, 2) for a in C]))
    #print("S: ", S)
    #print("C: ", C)
    return np.exp(-gamma_s*S) * np.exp(-gamma_c*C)

def distance_on_kernel(img, cluster, cluster_count):
    distanceOnKernel = np.zeros((100,100, len(cluster_count)))
    gramMatrix = np.zeros(len(cluster_count))
    newCluster = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            newCluster[i][j] = -1
    # compute k right terms
    # for i in range(100):
    #     for j in range(100):
    #         sx = [i, j]
    #         cx = [img[i][j][0], img[i][j][1], img[i][j][2]]
    #         for s in range(100):
    #             for t in range(100):
    #                 if cluster[i][j] == cluster[s][t]:
    #                     sx_apostrophe = [s, t]
    #                     cx_apostrophe = [img[s][t][0], img[s][t][1], img[s][t][2]]
    #                     gramMatrix[cluster[i][j]] += user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe)

    # apply Wei-chen Chiu's Unsupervised_Learning.pdf p.22 formula
    print("Computing formula...")
    for i in range(100):
        print("Computing row %d:" %i)
        for j in range(100):
            # left term
            sx = [i, j]
            cx = [img[i][j][0], img[i][j][1], img[i][j][2]]
            kxx = user_defined_kernel(sx, sx, cx, cx)
            for p in range(len(cluster_count)):
                distanceOnKernel[i][j][p] = kxx

            # middle term, gram matrix
            middleTerm = np.zeros(3)
            for s in range(100):
                for t in range(100):
                    sx_apostrophe = [s, t]
                    cx_apostrophe = [img[s][t][0], img[s][t][1], img[s][t][2]]
                    #print("cluster[s][t]", int(cluster[s][t]))
                    tmp = user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe)
                    #print("middleTerm, tmp:", middleTerm[int(cluster[s][t])], tmp)
                    middleTerm[int(cluster[s][t])] += tmp
                    if cluster[i][j] == cluster[s][t]:
                        gramMatrix[int(cluster[i][j])] += tmp

            for c in range(len(cluster_count)):
                middleTerm[c] = -2 * middleTerm[c] / cluster_count[int(cluster[i][j])]
                distanceOnKernel[i][j][c] += middleTerm[c]

    # right term
    rightTerm = np.copy(gramMatrix)
    for i in range(len(rightTerm)):
        rightTerm[i] = rightTerm[i] / cluster_count[i] / cluster_count[i]
    for i in range(100):
        for j in range(100):
            mini = np.Inf
            for c in range(len(cluster_count)):
                distanceOnKernel[i][j][c] += rightTerm[c]
                if distanceOnKernel[i][j][c] < mini:
                    mini = distanceOnKernel[i][j][c]
                    newCluster[i][j] = c
    return newCluster

def kernel_kmeans(img, k):
    # initialize centroid
    centroid = []
    cluster = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            cluster[i][j] = -1
    i = 0
    while i < k:
        tmp = np.random.randint(100, size=2)
        for j in range(i):
            while tmp[0] == centroid[j][0] and tmp[1] == centroid[j][1]:
                tmp = np.random.randint(100, size=2)
        centroid.append(tmp)
        cluster[tmp[0]][tmp[1]] = i
        i += 1
    centroid = np.array(centroid)
    print("Initial centorid: ", centroid)

    # kernel, compute distance
    K = np.zeros([100, 100])
    for i in range(100):
        for j in range(100):
            sx = [i, j]
            cx = [img[i][j][0], img[i][j][1], img[i][j][2]]
            K[i][j] = np.Inf
            for cen in range(k):
                sx_apostrophe = centroid[cen]
                cx_apostrophe = [img[centroid[cen][0]][centroid[cen][1]][0], \
                                 img[centroid[cen][0]][centroid[cen][1]][1], \
                                 img[centroid[cen][0]][centroid[cen][1]][2]]
                tmp = user_defined_kernel(sx, sx_apostrophe, cx, cx_apostrophe)
                if tmp < K[i][j]:
                    K[i][j] = tmp
                    cluster[i][j] = cluster[centroid[cen][0]][centroid[cen][1]]
                #print(K[i][j])
    cluster_count = np.zeros(k)
    for i in range(100):
        for j in range(100):
            for p in range(k):
                if p == cluster[i][j]:
                    cluster_count[p] += 1
                    break
    print("cluster_count: ", cluster_count)

    # recompute centroid
    newCluster = distance_on_kernel(img, cluster, cluster_count)
    for i in range(100):
        print("newCluster[%d]" %i)
        for j in range(100):
            print(newCluster[i][j])

    #####
    X0 = []
    Y0 = []
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(100):
        for j in range(100):
            if cluster[i][j] == 0:
                X0.append(99-i)
                Y0.append(j)
            elif cluster[i][j] == 1:
                X1.append(99-i)
                Y1.append(j)
            elif cluster[i][j] == 2:
                X2.append(99-i)
                Y2.append(j)
    plt.plot(X0, Y0, 'r.')
    plt.plot(X1, Y1, 'g.')
    plt.plot(X2, Y2, 'b.')

    #####
    X0 = []
    Y0 = []
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(100):
        for j in range(100):
            if newCluster[i][j] == 0:
                X0.append(99-i)
                Y0.append(j)
            elif newCluster[i][j] == 1:
                X1.append(99-i)
                Y1.append(j)
            elif newCluster[i][j] == 2:
                X2.append(99-i)
                Y2.append(j)
    plt.plot(X0, Y0, 'r.')
    plt.plot(X1, Y1, 'g.')
    plt.plot(X2, Y2, 'b.')
    plt.show()
    plt.show()

if __name__ == "__main__":
    im = imageio.imread('./image1.png')
    kernel_kmeans(im, 3)
    #user_defined_kernel([2,5], [9,9], [2,3,9], [9,2,4])
