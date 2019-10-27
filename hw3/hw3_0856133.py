import numpy as np
import matplotlib.pyplot as plt

# q1.a univariate gaussian data generator
# box-muller transform
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

# q1.b polynomial basis linear model data generator
def pblmdg(basis_number, sdv, weight):
    x = np.random.uniform(-1, 1, 1)
    epsilon = np.random.normal(0, sdv)
    #print("x: ", x, "epsilon: ", epsilon)
    y = 0
    for i in range(basis_number):
        y += weight[i]*np.power(x, i)
    y += epsilon
    return (x[0], y[0])

# Welford's online algorithm
def sequential_estimator(ugdg_mean, ugdg_var):
    count = 1
    old_sample_mean = ugdg(ugdg_mean,ugdg_var)
    old_sample_var = 1.
    print("initial mean/initial data: ", old_sample_mean, "initial var: ", old_sample_var)

    new_data = ugdg(ugdg_mean, ugdg_var)
    print("second data: ", new_data)
    count += 1
    new_sample_mean = old_sample_mean + (new_data - old_sample_mean) / count
    new_sample_var = old_sample_var + np.power((new_data - old_sample_mean), 2) / count - old_sample_var / (count - 1)
    print("second mean: ", new_sample_mean, "second var: ", new_sample_var)
    while np.abs(new_sample_mean - old_sample_mean) + np.abs(new_sample_var - old_sample_var) > 0.000001:
        old_sample_mean = new_sample_mean
        old_sample_var = new_sample_var

        new_data = ugdg(ugdg_mean, ugdg_var)
        print("Add data point: ", new_data)
        count += 1
        new_sample_mean = old_sample_mean + (new_data - old_sample_mean) / count
        new_sample_var = old_sample_var + np.power((new_data - old_sample_mean), 2) / count - old_sample_var / (count-1)
        print("Mean = ", new_sample_mean, "Variance = ", new_sample_var)
    return (new_sample_mean, new_sample_var)

# bayesian linear regression
# b : prior's covariance coefficient,
# inv(b) dot I : prior function's variance ,
# we assume that every dimension is independent to each other , so it's a diagonal matrix
# basis_number : if it's 2, then it's a line. if it's 3, then it's a parabola. and so on.
# a : likelihood function's variance, which is used to generate epsilon in pblmdg above
# weight : ground truth weight
# posterior mean : updated weight
def BLR(b, basis_number, a, weight):
    x, y = pblmdg(basis_number, a, weight)
    print("Add data point ", (x, y))
    x_array = np.array((x))
    y_array = np.array((y))
    X = np.zeros((1, basis_number))
    for i in range(basis_number):
        X[0][i] = np.power(x, i)
    #print("X = ", X)
    prior_variance = np.ones((basis_number, basis_number))
    prior_mean = np.zeros((basis_number, 1))
    posterior_variance = a * np.dot(np.transpose(X), X) + b * np.identity(basis_number)
    posterior_mean = a * np.dot(np.linalg.inv(posterior_variance), np.dot(np.transpose(X), y))
    print("posterior mean :")
    print(posterior_mean)
    print("posterior variance:")
    print(np.linalg.inv(posterior_variance))
    pred_mean = np.dot(X, posterior_mean)[0][0]
    pred_mean_array = np.array((pred_mean))
    pred_var = 1 / a + np.dot(np.dot(X, np.linalg.inv(posterior_variance)), np.transpose(X))[0][0]
    pred_var_array = np.array((pred_var))
    print("predictive distribution ~ N(", pred_mean, ",", pred_var, " )")
    print("==============================================")
    cnt = 1
    posterior_mean_10 = np.array((4, 1))
    posterior_mean_50 = np.array((4, 1))
    threshold = np.power(10., -8.)
    judge = threshold
    while judge >= threshold:
    #while np.abs(posterior_mean[0][0] - prior_mean[0][0] ) > 0.0001 and cnt < 10000:
    #while cnt < 3000:
        cnt += 1
        x, y = pblmdg(basis_number, a, weight)
        x_array = np.append(x_array, x)
        y_array = np.append(y_array, y)
        print("Add data point ", (x, y))
        for i in range(basis_number):
            X[0][i] = np.power(x, i)
        prior_variance = posterior_variance
        prior_mean = posterior_mean
        posterior_variance = a * np.dot(np.transpose(X), X) + prior_variance
        posterior_mean = \
            np.dot( np.linalg.inv(posterior_variance), (a*np.dot(np.transpose(X), y) + np.dot(prior_variance, prior_mean)))
        if cnt == 10:
            posterior_mean_10 = posterior_mean
            posterior_variance_10 = posterior_variance
        elif cnt == 50:
            posterior_mean_50 = posterior_mean
            posterior_variance_50 = posterior_variance
        print("posterior mean :")
        print(posterior_mean)
        print("posterior variance:")
        print(np.linalg.inv(posterior_variance))
        pred_mean = np.dot(X, posterior_mean)[0][0]
        pred_mean_array = np.append(pred_mean_array, pred_mean)
        pred_var = 1/a + np.dot(np.dot(X, np.linalg.inv(posterior_variance)), np.transpose(X))[0][0]
        pred_var_array = np.append(pred_var_array, pred_var)
        print("predictive distribution ~ N(", pred_mean, ",", pred_var, " )")
        print("==============================================")
        judge = 0
        for i in range(basis_number):
            judge += np.abs(posterior_mean[i][0] - prior_mean[i][0])
    #print(x_array.shape, cnt)
    #print(posterior_mean, pred_mean)


    # visualization
    x = np.linspace(-2, 2, 1000)
    def calculate_y(x, basis_number, weight):
        tmp = 0
        for i in range(basis_number):
            tmp += weight[i] * np.power(x, i)
        return tmp
    # def calculate_y_with_var(x, basis_number, weight, var):
    #     return
    plt.figure(figsize=(6, 6))
    my_string = 'b= ', 'basis_number= ', basis_number, 'a= ', a, 'weight= ', weight
    plt.title(my_string)
    plt.subplot(221)
    plt.title("ground truth")
    mean_of_red_line = np.zeros((len(x)))
    var_of_red_line = np.zeros((len(x)))
    for i in range(len(x)):
        mean_of_red_line[i] = calculate_y(x[i], basis_number, weight)
        X_for_red_line = np.zeros((1, basis_number))
        for j in range(basis_number):
            X_for_red_line[0][j] = np.power(x[i], j)
        #initial_var = a * np.dot(np.transpose(X), X) + b * np.identity(basis_number)
        #var_of_red_line[i] = 1/a + np.dot(np.dot(X_for_red_line, np.linalg.inv(initial_var)), np.transpose(X_for_red_line))
        var_of_red_line[i] = 1/a
    upper_y_final = mean_of_red_line + var_of_red_line
    lower_y_final = mean_of_red_line - var_of_red_line
    plt.plot(x, calculate_y(x, basis_number, weight), 'black')
    #plt.plot(x_array, y_array, 'bo')
    plt.plot(x, upper_y_final, 'red')
    plt.plot(x, lower_y_final, 'red')
    plt.axis([-2, 2, -20, 20])

    plt.subplot(222)
    plt.title("predict")
    mean_of_red_line = np.zeros((len(x)))
    var_of_red_line = np.zeros((len(x)))
    for i in range(len(x)):
        mean_of_red_line[i] = calculate_y(x[i], basis_number, posterior_mean)
        X_for_red_line = np.zeros((1, basis_number))
        for j in range(basis_number):
            X_for_red_line[0][j] = np.power(x[i], j)
        var_of_red_line[i] = 1/a + np.dot(np.dot(X_for_red_line, np.linalg.inv(posterior_variance)), np.transpose(X_for_red_line))
    upper_y_final = mean_of_red_line + var_of_red_line
    lower_y_final = mean_of_red_line - var_of_red_line
    # print(x.shape)
    # print(mean_of_red_line.shape)
    # print(var_of_red_line.shape)
    plt.plot(x, calculate_y(x, basis_number, posterior_mean),'black')
    plt.plot(x_array, y_array, 'bo')
    plt.plot(x, upper_y_final, 'red')
    plt.plot(x, lower_y_final, 'red')
    plt.axis([-2, 2, -20, 20])

    plt.subplot(223)
    plt.title("After 10 incomes")
    mean_of_red_line_10 = np.zeros((len(x)))
    var_of_red_line_10 = np.zeros((len(x)))
    for i in range(len(x)):
        mean_of_red_line_10[i] = calculate_y(x[i], basis_number, posterior_mean_10)
        X_for_red_line = np.zeros((1, basis_number))
        for j in range(basis_number):
            X_for_red_line[0][j] = np.power(x[i], j)
        var_of_red_line_10[i] = 1 / a + np.dot(np.dot(X_for_red_line, np.linalg.inv(posterior_variance_10)),
                                            np.transpose(X_for_red_line))
    upper_y_final_10 = mean_of_red_line_10 + var_of_red_line_10
    lower_y_final_10 = mean_of_red_line_10 - var_of_red_line_10
    plt.plot(x, calculate_y(x, basis_number, posterior_mean_10), 'black')
    plt.plot(x_array[:10], y_array[:10], 'bo')
    plt.plot(x, upper_y_final_10, 'red')
    plt.plot(x, lower_y_final_10, 'red')
    plt.axis([-2, 2, -20, 20])

    plt.subplot(224)
    plt.title("After 50 incomes")
    mean_of_red_line_50 = np.zeros((len(x)))
    var_of_red_line_50 = np.zeros((len(x)))
    for i in range(len(x)):
        mean_of_red_line_50[i] = calculate_y(x[i], basis_number, posterior_mean_50)
        X_for_red_line = np.zeros((1, basis_number))
        for j in range(basis_number):
            X_for_red_line[0][j] = np.power(x[i], j)
        var_of_red_line_50[i] = 1 / a + np.dot(np.dot(X_for_red_line, np.linalg.inv(posterior_variance_50)),
                                            np.transpose(X_for_red_line))
    upper_y_final_50 = mean_of_red_line_50 + var_of_red_line_50
    lower_y_final_50 = mean_of_red_line_50 - var_of_red_line_50
    plt.plot(x, calculate_y(x, basis_number, posterior_mean_50), 'black')
    plt.plot(x_array[:50], y_array[:50], 'bo')
    plt.plot(x, upper_y_final_50, 'red')
    plt.plot(x, lower_y_final_50, 'red')
    plt.axis([-2, 2, -20, 20])
    plt.show()

if __name__ == '__main__' :
    #sample_mean, sample_var =  sequential_estimator(3,5)
    #BLR(1, 4, 1, [1, 2, 3, 4])
    BLR(100, 4, 1, [1, 2, 3, 4])
    #BLR(1, 3, 3, [1, 2, 3])