import numpy as np
import pandas


def ugdg(mean, var):
    U1 = np.random.uniform(0, 1, 1)
    U2 = np.random.uniform(0, 1, 1)
    Z0 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)
    Z1 = np.sqrt(-2*np.log(U1)) * np.sin(2*np.pi*U2)
    #print("U1: ", U1, "\nZ0: ", Z0)
    #print("U2: ", U2, "\nZ1: ", Z1)
    Z0 = Z0*np.sqrt(var) + mean
    Z1 = Z1*np.sqrt(var) + mean
    #return (Z0[0], Z1[0])
    return Z0[0]


def sigmoid(x1, x2, theta):
    X = np.zeros(3)
    X[0] = 1
    X[1] = x1
    X[2] = x2
    z = np.dot(np.transpose(theta), X)
    z = np.minimum(z, 20)
    z = np.maximum(z, -20)
    sig = 1 / (1 + np.exp(-z))
    #sig = np.minimum(sig, 0.9999)
    #sig = np.maximum(sig, 0.0001)
    return sig


def log_likelihood(x1, x2, label, theta):
    sigmoid_probs = np.zeros(len(x1))
    l = 0
    for i in range(len(x1)):
        sigmoid_probs[i] = sigmoid(x1[i], x2[i], theta)
        l += (label[i]*np.log(sigmoid_probs[i]) + (1-label[i])*np.log(1 - sigmoid_probs[i]))
    return l


def gradient(x1, x2, label, theta):
    sigmoid_probs = np.zeros(len(x1))
    g = np.zeros(3)
    for i in range(len(x1)):
        sigmoid_probs[i] = sigmoid(x1[i], x2[i], theta)
        g[0] += (label[i] - sigmoid_probs[i])*1
        g[1] += (label[i] - sigmoid_probs[i])*x1[i]
        g[2] += (label[i] - sigmoid_probs[i])*x2[i]
    return g


def hessian(x1, x2, label, theta):
    sigmoid_probs = np.zeros(len(x1))
    A = np.zeros((len(x1), len(theta)))
    D = np.zeros((len(x1), len(x1)))
    for i in range(len(x1)):
        sigmoid_probs[i] = sigmoid(x1[i], x2[i], theta)
        A[i][0] = 1
        A[i][1] = x1[i]
        A[i][2] = x2[i]
        D[i][i] = sigmoid_probs[i]*(1 - sigmoid_probs[i])
    AT = np.transpose(A)
    return np.dot(np.dot(AT, D), A)
    # sigmoid_probs = np.zeros(len(x1))
    # hess = np.zeros((3, 3))
    # for i in range(len(x1)):
    #     sigmoid_probs[i] = sigmoid(x1[i], x2[i], theta)
    #     hess[0][0] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*1
    #     hess[0][1] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x1[i]
    #     hess[0][2] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x2[i]
    #     hess[1][0] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x1[i]
    #     hess[1][1] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x1[i]*x1[i]
    #     hess[1][2] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x1[i]*x2[i]
    #     hess[2][0] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x2[i]
    #     hess[2][1] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x1[i]*x2[i]
    #     hess[2][2] += -sigmoid_probs[i]*(1 - sigmoid_probs[i])*x2[i]*x2[i]
    # return hess


def newton_method(x1, x2, label, theta):
    print("====================BELOW: NEWTON's METHOD====================")
    threshold = 0.01
    new_theta = np.zeros(3)
    new_theta[0] = 100
    new_theta[1] = 100
    new_theta[2] = 100
    judge = np.abs((new_theta[0] - theta[0])) + np.abs((new_theta[1] - theta[1])) + np.abs((new_theta[2] - theta[2]))
    while judge >= threshold:
        # DERIVATIVES
        g = gradient(x1, x2, label, theta)
        hess = hessian(x1, x2, label, theta)
        print("hess: ", hess)
        if np.linalg.matrix_rank(hess) != len(theta):
            print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSingular:")
            lr = 0.0001
            new_theta = theta + lr * g
        else:
            print("Non-singular:")
            print("g: ", g)
            H_inv = np.linalg.inv(hess)
            print("H_inv: ", H_inv)
            g = np.expand_dims(g, axis=1)
            delta = np.dot(H_inv, g)
            delta = np.squeeze(delta, axis=1)
            print(H_inv.shape, g.shape, delta.shape)
            new_theta = theta - delta
        judge = np.abs((new_theta[0] - theta[0])) + np.abs((new_theta[1] - theta[1])) + np.abs((new_theta[2] - theta[2]))
        theta = new_theta
        print("my theta: ", theta)
        print("log_likelihood: ", log_likelihood(x1, x2, label, theta))
    pred_label = predict_label(x1, x2, label, theta)
    print("====================ABOVE: NEWTON's METHOD====================")


def gradient_ascent(x1, x2, label, theta):
    print("====================BELOW: GRADIENT ASCENT====================")
    lr = 0.0001
    threshold = 0.01
    new_theta = np.zeros(3)
    new_theta[0] = 100
    new_theta[1] = 100
    new_theta[2] = 100
    judge = np.abs((new_theta[0] - theta[0])) + np.abs((new_theta[1] - theta[1])) + np.abs((new_theta[2] - theta[2]))
    while judge >= threshold:
        # GRADIENT
        g = gradient(x1, x2, label, theta)
        new_theta = theta + lr*g
        judge = np.abs((new_theta[0] - theta[0])) + np.abs((new_theta[1] - theta[1])) + np.abs((new_theta[2] - theta[2]))
        theta = new_theta
        print("my theta: ", theta)
        print("log_likelihood: ", log_likelihood(x1, x2, label, theta))
    pred_label = predict_label(x1, x2, label, theta)
    print("====================ABOVE: GRADIENT ASCENT====================")


def predict_label(x1, x2, label, theta):
    pred_label = np.zeros(len(label))
    for i in range(len(x1)):
        pred_label[i] = -1
        judge = sigmoid(x1[i], x2[i], theta)
        if judge >= .5 and judge <= 1:
            pred_label[i] = 1
        elif judge < .5 and judge >= 0:
            pred_label[i] = 0
        else:
            pred_label[i] = -100
    print("DF_data['label']: \n", label)
    print("pred_label: \n", pred_label)
    same = 0
    for i in range(len(label)):
        if label[i] == pred_label[i]:
            same += 1
    print("Same = %d, total = %d , Accuracy = %f " %(same, len(label), same / len(label)))
    return pred_label


def logistic_regression():
    # input
    n = int(input('number of data points: '))
    mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = [int(x) for x in input('enter mean, var: ').split()]

    # initialization
    ## dataset d1,d2
    ## column0 = x, column1 = y, column2 = label
    d1, d2 = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        d1[i][0] = ugdg(mx1, vx1)
        d1[i][1] = ugdg(my1, vy1)
        d1[i][2] = 1
        d2[i][0] = ugdg(mx2, vx2)
        d2[i][1] = ugdg(my2, vy2)
        d2[i][2] = 0
    df_data = np.concatenate((d1, d2), axis=0)
    df_data = pandas.DataFrame(data=df_data, columns=['x1', 'x2', 'label'])
    print(df_data)

    # gradient ascent
    ## theta: weight to multiply x
    theta = np.ones(3)  # shape (3,)
    print("my theta: ", theta)
    # log_likelihood
    l = log_likelihood(df_data['x1'], df_data['x2'], df_data['label'], theta)
    print("log_likelihood: ", l)
    gradient_ascent(df_data['x1'], df_data['x2'], df_data['label'], theta)

    # newton's method
    ## theta: weight to multiply x
    theta = np.ones(3)  # shape (3,)
    print("my theta: ", theta)
    # log_likelihood
    l = log_likelihood(df_data['x1'], df_data['x2'], df_data['label'], theta)
    print("log_likelihood: ", l)
    newton_method(df_data['x1'], df_data['x2'], df_data['label'], theta)

if __name__ == '__main__':
    logistic_regression()
