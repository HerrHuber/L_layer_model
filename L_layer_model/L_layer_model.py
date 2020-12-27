# -*- coding: utf-8 -*-
# This Program

import time
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from PIL import Image


"""
L layer binary classification model
Images (64, 64, 3)
Input: n_x = 12288
L hidden Layers: n_lx (x = 1 .. L-1)
Output: n_yhay = 1

"""

# load data
# two layers
# init parameters
# for i in range(iterations):
# foreward propagation
# backward propagation
# train accuracy
# test accuracy


# load data
def load_data(filename):
    """
    Dataset format should is h5 and look like this:
    datasets/
    --filename
    ----train_x (m, 64, 64, 3)
    ----train_y (m,)

    m := number of training examples
    filename := e.g. "mydataset.h5"
    """
    dataset = h5py.File(str(filename), "r")
    X = np.array(dataset["train_x"][:])
    Y = np.array(dataset["train_y"][:])
    # reshape from (m,) to (1, m)
    Y = Y.reshape((1, Y.shape[0]))
    return X, Y


# two layers
# init parameters
def init_params(n_x, n_l1, n_yhat, random_on=False, seed=1):
    """
    Argument:
    n_x: number of input layer nodes
    n_l1: number of nodes layer 1
    n_yhat: number of nodes output layer
    random_on: use random seed
    seed: specify random seed
    """
    if(random_on):
        np.random.seed(seed)

    params = {
        "W1": np.random.standard_normal((n_l1, n_x)) * 0.01,
        "b1": np.zeros((n_l1, 1)),
        "W2": np.random.standard_normal((n_yhat, n_l1)) * 0.01,
        "b2": np.zeros((n_yhat, 1))
    }

    return params


# L layers
# init parameters
def L_init_params(layer_dims, random_on=False, seed=1):
    """
    Argument:
    n_x: number of input layer nodes
    n_l1: number of nodes layer 1
    n_yhat: number of nodes output layer
    random_on: use random seed
    seed: specify random seed
    """
    if(random_on):
        np.random.seed(seed)

    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)


def preprocess(X):
    # The "-1" makes reshape flatten the remaining dimensions
    X_flatten = X.reshape(X.shape[0], -1).T
    # Standardize data to have values between 0 and 1
    return X_flatten / 255.


def linear_forward(A, W, b):
    return W.dot(A) + b


def linear_activation_forward(A0, W, b, activation):
    if(activation == "sigmoid"):
        Z = linear_forward(A0, W, b)
        A1 = sigmoid(Z)
    elif activation == "relu":
        Z = linear_forward(A0, W, b)
        A1 = relu(Z)
    else:
        raise Exception("Possible activation functions \"sigmoid\" or \"relu\"")

    return A1, Z


def L_linear_activation_forward(X, params):
    Zs = []
    As = []
    Ws = []
    Al = X
    L = int(len(params) / 2)

    ##### Be carefull l != L #####
    # L-1 times linear->relu
    for l in range(1, L):
        Al_1 = Al
        Al, Z = linear_activation_forward(Al_1, params["W" + str(l)], params["b" + str(l)],
                                             activation="relu")
        Zs.append(Z)
        As.append(Al_1)
        Ws.append(params["W" + str(l)])

    AL, Z = linear_activation_forward(Al, params["W" + str(L)], params['b' + str(L)],
                                      activation="sigmoid")
    Zs.append(Z)
    As.append(AL)
    Ws.append(params["W" + str(L)])

    return AL, Zs, As, Ws


def compute_cost(A, Y):
    m = Y.shape[1]
    # Compute loss from A and y.
    cost = (1. / m) * (-np.dot(Y, np.log(A).T) - np.dot(1 - Y, np.log(1 - A).T))
    cost = np.squeeze(cost)  # turns [[42]] into 42
    return cost


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def sigmoid_backward_reg(AL, Y):
    return AL - Y


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def relu_backward_reg(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def linear_backward(dZ, A, W):
    m = A.shape[1]

    dW = 1. / m * np.dot(dZ, A.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA = np.dot(W.T, dZ)

    return dA, dW, db


def linear_activation_backward(dA1, Z, A, W, activation):
    if(activation == "relu"):
        dZ = relu_backward(dA1, Z)
        dA0, dW, db = linear_backward(dZ, A, W)
    elif(activation == "sigmoid"):
        dZ = sigmoid_backward(dA1, Z)
        dA0, dW, db = linear_backward(dZ, A, W)
    else:
        raise Exception("Possible activation functions \"sigmoid\" or \"relu\"")

    return dA0, dW, db


def L_linear_activation_backward(AL, Y, Z, A, W):
    grads = {}
    L = len(Z)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # initialize backward propagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # sigmoid->linear
    ZL = Z[L - 1]
    AL = A[L - 1]
    WL = W[L - 1]
    dAL_1, dWL, dbL = linear_activation_backward(dAL, ZL, AL, WL, activation="sigmoid")
    grads["dA" + str(L - 1)] = dAL_1
    grads["dW" + str(L)] = dWL
    grads["db" + str(L)] = dbL

    # L-1 times relu->linear
    for l in reversed(range(L - 1)):
        Zl = Z[l]
        Al = A[l]
        Wl = W[l]
        dAl_1, dW, db = linear_activation_backward(grads["dA" + str(l + 1)],
                                                   Zl, Al, Wl, activation="relu")
        grads["dA" + str(l)] = dAl_1
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_params(params, grads, learning_rate):
    # 2 = number of layers in the neural network
    for l in range(2):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return params


def L_update_params(params, grads, learning_rate):
    L = int(len(params) / 2)
    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return params


def two_layer_forward(X, params):
    A0, Z = linear_activation_forward(X, params["W1"],
                                         params["b1"], "relu")
    A1, Z = linear_activation_forward(A0, params["W2"],
                                          params["b2"], "sigmoid")
    return A1


#def L_layer_forward(X, params):
#    return L_linear_activation_forward(X, params)


def predict(X, params):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probabilities = two_layer_forward(X, params)

    # convert probabilities to 0/1 predictions
    for i in range(probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def L_predict(X, params):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    #probabilities = L_layer_forward(X, params)
    probabilities, Zs, As, Ws = L_linear_activation_forward(X, params)

    # convert probabilities to 0/1 predictions
    for i in range(probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def accuracy(p, Y):
    m = Y.shape[1]
    return np.sum((p == Y)) / m


def two_layer_model_continue(X, Y, params, iterations,
                             learning_rate, print_on, plot_on):
    grads = {}
    costs = []
    for i in range(iterations):
        # foreward propagation
        A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
        A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
        # compute cost
        cost = compute_cost(A2, Y)
        # backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, Z2, A1, params["W2"], "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, Z1, X, params["W1"], "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        # update params
        params = update_params(params, grads, learning_rate)

        # print cost
        if(print_on):
            if(i % 100 == 0):
                print("Cost after iteration", str(i) + ":", np.squeeze(cost))
        if(i % 10 == 0):
            costs.append(float(cost))

    # plot cost
    if(plot_on):
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return params, costs


def L_layer_model_continue(X, Y, params, iterations,
                           learning_rate, print_on, plot_on):
    costs = []
    for i in range(iterations):
        # foreward propagation
        AL, Z, A, W = L_linear_activation_forward(X, params)
        # compute cost
        cost = compute_cost(AL, Y)
        # backward propagation
        grads = L_linear_activation_backward(AL, Y, Z, A, W)
        # update params
        params = L_update_params(params, grads, learning_rate)

        # print cost
        if(print_on):
            if(i % 100 == 0):
                print("Cost after iteration", str(i) + ":", np.squeeze(cost))
        if(i % 10 == 0):
            costs.append(float(cost))

    # plot cost
    if(plot_on):
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return params, costs


def two_layer_model(X, Y, layer_dims, random_on, seed, iterations,
                    learning_rate, print_on, plot_on):
    n_x, n_l1, n_yhat = layer_dims
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)
    return two_layer_model_continue(X, Y, params, iterations,
                                    learning_rate, print_on, plot_on)


def L_layer_model(X, Y, layer_dims, random_on, seed, iterations,
                  learning_rate, print_on, plot_on):
    params = L_init_params(layer_dims, random_on, seed)
    return L_layer_model_continue(X, Y, params, iterations,
                                  learning_rate, print_on, plot_on)


def init_adam(parameters):
    L = int(len(parameters) / 2)
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # partition (except end case)
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def update_params_adam(params, grads, v, s, t, learning_rate=0.01,
                       beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = int(len(params) / 2)
    v_corrected = {}
    s_corrected = {}

    # adam update
    for l in range(L):
        # moving average of gradients
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # compute bias corrected first moment estimate
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        # moving average of squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (grads["dW" + str(l + 1)] ** 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (grads["db" + str(l + 1)] ** 2)

        # compute bias corrected second raw moment estimate
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        # Update parameters
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * v_corrected[
            "dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * v_corrected[
            "db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

    return params, v, s


def L_layer_model_opt_continue(X, Y, params, seed, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
                      beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_on=True, plot_on=False):
    costs = []
    t = 0  # counter for Adam

    v, s = init_adam(params)

    # optimization loop
    for i in range(num_epochs):
        # increment seed to reshuffle differently after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost = 0

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # forward propagation
            AL, Z, A, W = L_linear_activation_forward(minibatch_X, params)

            # compute cost
            cost += compute_cost(AL, minibatch_Y)

            # backward propagation
            grads = L_linear_activation_backward(AL, minibatch_Y, Z, A, W)

            # update parameters
            t = t + 1  # Adam counter
            params, v, s = update_params_adam(params, grads, v, s,
                                              t, learning_rate, beta1, beta2, epsilon)

        if(print_on and i % 100 == 0):
            print("Cost after epoch", str(i) + ":", np.squeeze(cost))
        if(i % 10 == 0):
            costs.append(float(cost))

    if(plot_on):
        plt.plot(np.squeeze(costs))
        plt.ylabel("Cost")
        plt.xlabel("epochs (per 10)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return params, costs


def L_layer_model_opt(X, Y, layer_dims, random_on, seed, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
                      beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_on=True, plot_on=False):
    params = L_init_params(layer_dims, random_on, seed)
    return L_layer_model_opt_continue(X, Y, params, seed, learning_rate, mini_batch_size, beta,
                                      beta1, beta2, epsilon, num_epochs, print_on, plot_on)


def classify(imagename, params):
    image = Image.open(imagename)
    image = image.resize((64, 64))
    img = np.array(image)
    plt.imshow(img)
    plt.show()
    img = preprocess(np.array([img]))

    # maybe use np.squeeze
    return predict(img, params)[0]


def L_classify(imagename, params):
    image = Image.open(imagename)
    image = image.resize((64, 64))
    img = np.array(image)
    plt.imshow(img)
    plt.show()
    img = preprocess(np.array([img]))

    return L_predict(img, params)[0]


def save_params(params, filename):
    np.savez(filename, W1=params["W1"],
            b1=params["b1"], W2=params["W2"], b2=params["b2"])


def L_save_params(params, filename):
    np.savez(filename, **params)


def load_params(filename):
    new = np.load(filename)
    newparams = {}
    for i in new.files:
        newparams[i] = new[i]

    return newparams


def L_load_params(filename):
    return load_params(filename)


def L_layer_model_example():
    filename = "../datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    layer_dims = [X.shape[0], 20, 10, 1]
    random_on = True
    seed = 1

    iterations = 100
    learning_rate = 0.001

    print_on = True
    plot_on = True

    params, costs = L_layer_model(X, Y, layer_dims, random_on, seed, iterations,
                                  learning_rate, print_on, plot_on)

    print("params.keys(): ", params.keys())
    # train accuracy
    p = L_predict(X, params)
    print("Train accuracy: ", accuracy(p, Y))
    # test accuracy
    test_filename = "../datasets/train_catvnoncat.h5"
    test_dataset = h5py.File(test_filename, "r")
    X_test = np.array(test_dataset["train_set_x"][:])
    Y_test = np.array(test_dataset["train_set_y"][:])
    # reshape from (m,) to (1, m)
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    X_test = preprocess(X_test)
    p_test = L_predict(X_test, params)
    print("Test accuracy: ", accuracy(p_test, Y_test))

    # predict new image with pretrained parameters
    imagename = "../images/Image.jpg"
    prediction = L_classify(imagename, params)
    print("The model predicts: ", prediction)

    print()
    print("Save params")
    print("params.keys: ", params.keys())
    paramsfilename = "../datasets/catvnoncat_params_2.npz"
    L_save_params(params, paramsfilename)

    newparams = L_load_params(paramsfilename)

    print("Accuracy of model with loaded parameters")
    # train accuracy
    p = L_predict(X, newparams)
    print("Train accuracy: ", accuracy(p, Y))
    p_test = L_predict(X_test, newparams)
    print("Test accuracy: ", accuracy(p_test, Y_test))

    newnewparams, newnewcosts = L_layer_model_continue(X, Y, newparams, iterations,
                                                         learning_rate, print_on, plot_on)

    #print("newnewcosts: ", newnewcosts)

    print("Accuracy of model with pretrained and additionally trained parameters")
    # train accuracy
    p = L_predict(X, newnewparams)
    print("Train accuracy: ", accuracy(p, Y))
    p_test = L_predict(X_test, newnewparams)
    print("Test accuracy: ", accuracy(p_test, Y_test))


def main():
    filename = "../datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    layer_dims = [X.shape[0], 10, 1]
    random_on = True
    seed = 1

    learning_rate = 0.0003
    mini_batch_size = 64
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    num_epochs = 100

    print_on = True
    plot_on = True

    params, costs = L_layer_model_opt(X, Y, layer_dims, random_on, seed, learning_rate, mini_batch_size,
                                      beta, beta1, beta2, epsilon, num_epochs, print_on, plot_on)

    print("params.keys(): ", params.keys())
    # train accuracy
    p = L_predict(X, params)
    print("Train accuracy: ", accuracy(p, Y))
    # test accuracy
    test_filename = "../datasets/train_catvnoncat.h5"
    test_dataset = h5py.File(test_filename, "r")
    X_test = np.array(test_dataset["train_set_x"][:])
    Y_test = np.array(test_dataset["train_set_y"][:])
    # reshape from (m,) to (1, m)
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    X_test = preprocess(X_test)
    p_test = L_predict(X_test, params)
    print("Test accuracy: ", accuracy(p_test, Y_test))

    # predict new image with pretrained parameters
    imagename = "../images/Image.jpg"
    prediction = L_classify(imagename, params)
    print("The model predicts: ", prediction)

    print()
    print("Save params")
    print("params.keys: ", params.keys())
    paramsfilename = "../datasets/catvnoncat_params_2.npz"
    L_save_params(params, paramsfilename)

    newparams = L_load_params(paramsfilename)

    print("Accuracy of model with loaded parameters")
    # train accuracy
    p = L_predict(X, newparams)
    print("Train accuracy: ", accuracy(p, Y))
    p_test = L_predict(X_test, newparams)
    print("Test accuracy: ", accuracy(p_test, Y_test))

    newnewparams, newnewcosts = L_layer_model_opt_continue(X, Y, newparams, seed, learning_rate, mini_batch_size,
                                      beta, beta1, beta2, epsilon, num_epochs, print_on, plot_on)

    #print("newnewcosts: ", newnewcosts)

    print("Accuracy of model with pretrained and additionally trained parameters")
    # train accuracy
    p = L_predict(X, newnewparams)
    print("Train accuracy: ", accuracy(p, Y))
    p_test = L_predict(X_test, newnewparams)
    print("Test accuracy: ", accuracy(p_test, Y_test))


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#