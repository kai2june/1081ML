import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def rational_quadratic_kernel(xa, xb, sigmaf = 1, alpha = 1, l = 1):
    # sigmaf = 1.31
    # alpha = 423.11
    # l = 3.31
    numerator = np.abs(xa-xb)*np.abs(xa-xb)
    denominator = 2*alpha*l*l
    return sigmaf*sigmaf*np.power(1+numerator/denominator, -alpha)

def generate_x_star():
    x_star = np.random.uniform(-60, 60, 1)
    return x_star

def compute_K(X, Y, sigmaf=1, alpha=1, l=1):
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i][j] = rational_quadratic_kernel(X[i], X[j], sigmaf, alpha, l)
            if i == j:
                K[i][j] += 0.2
    return K

def compute_K_star(X, Y, x_star, sigmaf=1, alpha=1, l=1):
    K_star = np.zeros(len(X))
    for i in range(34):
        K_star[i] = rational_quadratic_kernel(x_star, X[i], sigmaf, alpha, l)
    return K_star

def GPR():
    # load file
    X = np.zeros(34)
    Y = np.zeros(34)
    i = 0
    input_file = open("./input.data", "r")
    for line in input_file.readlines():
        X[i], Y[i] = line.split(' ')
        print(X[i], Y[i])
        i += 1

    # kernel
    sample_count = 100
    K = compute_K(X, Y)
    X_lin = np.linspace(-60, 60, num=sample_count)
    Y_lin = np.zeros(sample_count)
    var_y_star = np.zeros(sample_count)
    interval_upper = np.zeros(sample_count)
    interval_lower = np.zeros(sample_count)
    for i in range(len(X_lin)):
        K_star = compute_K_star(X, Y, X_lin[i])
        K_star_star = rational_quadratic_kernel(X_lin[i], X_lin[i])
        y_star_bar = K_star@np.linalg.inv(K)@Y
        Y_lin[i] = y_star_bar
        var_y_star[i] = K_star_star - K_star@np.linalg.inv(K)@np.transpose(K_star)
    for i in range(sample_count):
        interval_upper[i] = Y_lin[i] + 1.96*np.sqrt(var_y_star[i])
        interval_lower[i] = Y_lin[i] - 1.96*np.sqrt(var_y_star[i])

    # visualization
    plt.title("sigmaf = 1, alpha = 1, l = 1")
    plt.plot(X, Y, 'b.')
    plt.plot(X_lin, Y_lin, 'black')
    plt.fill_between(X_lin, interval_upper, interval_lower)
    plt.plot(X_lin, Y_lin + 1.96*np.sqrt(var_y_star), 'green')
    plt.plot(X_lin, Y_lin - 1.96*np.sqrt(var_y_star), 'orange')
    plt.show()

    # q1_2 #fun = lambda C, sigmaf, alpha, l: (0.5*np.log(C)) + (0.5*Y.T @ C.inv @ Y) + (len(Y)/2)*np.log(np.pi)
    def C(xa, xb, sigmaf, alpha, l):
        numerator = np.abs(xa - xb) * np.abs(xa - xb)
        denominator = 2 * alpha * l * l
        return sigmaf * sigmaf * np.power(1 + numerator / denominator, -alpha)

    def fun(x, *args):
        # for arg in args:
        #     print("arg {}")
        #print("args {}".format(args))
        X, Y = args
        print("\nX {}\nY {}".format(X, Y))
        sigmaf, alpha, l = x
        print("sigmaf {} alpha {} l {}".format(sigmaf, alpha, l))
        K = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                K[i][j] = C(X[i], X[j], sigmaf, alpha, l)
                if i == j:
                    K[i][j] += 0.2
        return (0.5*np.log(np.linalg.det(K))) + (0.5*Y.T @ np.linalg.inv(K) @ Y) + (0.5*len(Y)*np.log(np.pi))
    res = minimize(fun, x0=[1, 1, 1], args=(X, Y))
    print("res: {}".format(res.x))

    # kernel
    sample_count = 100
    K = compute_K(X, Y, res.x[0], res.x[1], res.x[2])
    X_lin = np.linspace(-60, 60, num=sample_count)
    Y_lin = np.zeros(sample_count)
    var_y_star = np.zeros(sample_count)
    interval_upper = np.zeros(sample_count)
    interval_lower = np.zeros(sample_count)
    for i in range(len(X_lin)):
        K_star = compute_K_star(X, Y, X_lin[i], res.x[0], res.x[1], res.x[2])
        K_star_star = rational_quadratic_kernel(X_lin[i], X_lin[i], res.x[0], res.x[1], res.x[2])
        y_star_bar = K_star@np.linalg.inv(K)@Y
        Y_lin[i] = y_star_bar
        var_y_star[i] = K_star_star - K_star@np.linalg.inv(K)@np.transpose(K_star)
    for i in range(sample_count):
        interval_upper[i] = Y_lin[i] + 1.96*np.sqrt(var_y_star[i])
        interval_lower[i] = Y_lin[i] - 1.96*np.sqrt(var_y_star[i])

    # visualization
    plt.title("sigmaf = %0.4f, alpha = %0.4f, l = %0.4f" %(res.x[0], res.x[1], res.x[2]))
    plt.plot(X, Y, 'b.')
    plt.plot(X_lin, Y_lin, 'black')
    plt.fill_between(X_lin, interval_upper, interval_lower)
    plt.plot(X_lin, Y_lin + 1.96*np.sqrt(var_y_star), 'green')
    plt.plot(X_lin, Y_lin - 1.96*np.sqrt(var_y_star), 'orange')
    plt.show()

if __name__ == '__main__':
    GPR()