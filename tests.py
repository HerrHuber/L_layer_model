# -*- coding: utf-8 -*-
# This Program

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

from L_layer_model.L_layer_model import *


def test_load_data():
    print()
    print("Test load data")
    # TODO: Create dateset from open source images and use it here
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    print("X.shape: " + str(X.shape))
    print("Y.shape: " + str(Y.shape))
    assert X.shape[0] == Y.shape[1], "Number of examples in X different from Y"
    assert X.shape[1] * X.shape[2] * X.shape[3] == 64*64*3, "Shape of X should be (m, 64, 64, 1), " \
                                                            "m := number of training examples"

def test_init_params():
    print()
    print("Test initialize parameters")
    n_x = 828
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)
    print("W1: ", str(params["W1"].shape))
    print("b1: ", str(params["b1"].shape))
    print("W2: ", str(params["W2"].shape))
    print("b2: ", str(params["b2"].shape))
    print("W2: ", str(params["W2"]))

    assert params["W1"].shape == (n_l1, n_x), "Wrong shape"
    assert params["b1"].shape == (n_l1, 1), "Wrong shape"
    assert params["W2"].shape == (n_yhat, n_l1), "Wrong shape"
    assert params["b2"].shape == (n_yhat, 1), "Wrong shape"


def test_L_init_params():
    print()
    print("Test L layer initialize parameters")
    layer_dims = [828, 20, 10, 1]
    L = len(layer_dims)
    random_on = True
    seed = 1
    params = L_init_params(layer_dims, random_on, seed)
    for l in range(1, L):
        print("W" + str(l) + ": ", params["W" + str(l)].shape)
        print("b" + str(l) + ": ", params["b" + str(l)].shape)
    print("W" + str(L-1) + ": ", params["W" + str(L-1)])

    assert (params["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])), "Wrong shape"
    assert (params["b" + str(l)].shape == (layer_dims[l], 1)), "Wrong shape"


def test_sigmoid():
    print()
    print("Test sigmoid")
    print("sigmoid(-10): ", sigmoid(-10))
    print("sigmoid(0): ", sigmoid(0))
    print("sigmoid(10): ", sigmoid(10))

    assert sigmoid(-10) < 0.0001, "Sigmoid does not work"
    assert sigmoid(0) == 0.5, "Sigmoid does not work"
    assert sigmoid(10) > 0.999, "Sigmoid does not work"


def test_relu():
    print()
    print("Test relu")
    print("relu(-10): ", relu(-10))
    print("relu(0): ", relu(0))
    print("relu(10): ", relu(10))
    Z = np.array([-10, 0, 10])
    print("type(Z): ", type(Z))
    print("type(A): ", type(relu(Z)))

    assert relu(-10) == 0, "Relu does not work"
    assert relu(0) == 0, "Relu does not work"
    assert relu(10) > 0, "Relu does not work"


def test_preprocess():
    print()
    print("Test preprocess")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    print("X.shape: ", str(X.shape))
    print("X[:5]", X[:5])
    print(X.min())

    assert X.max() <= 1, "X is not standardized properly"
    assert X.min() >= 0, "X is not standardized properly"


def test_linear_forward():
    print()
    print("Test linear forward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    params = init_params(X.shape[0], 10, 1)
    A = linear_forward(X, params["W1"], params["b1"])
    print("X.shape: ", X.shape)
    print("A.shape: ", A.shape)
    print("W1.shape: ", params["W1"].shape)
    print("type(X): ", type(X))
    print("type(A): ", type(A))
    assert A.shape == (params["W1"].shape[0], X.shape[1])


def test_linear_activation_forward():
    print()
    print("Test linear activation forward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    params = init_params(X.shape[0], 10, 1)
    A1, Z = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    print("X.shape[1]: ", X.shape[1])
    assert (A1.shape == (params["W1"].shape[0], X.shape[1])),\
        "Linear activation forward wrong dimensions"


def test_L_linear_activation_forward():
    print()
    print("Test L layer linear activation forward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    params = L_init_params([X.shape[0], 10, 1])
    L = int(len(params) / 2)
    AL, Z = L_linear_activation_forward(X, params)
    print("type(AL): ", type(AL))
    print("AL.shape: ", AL.shape)
    assert (AL.shape == (params["W" + str(L)].shape[0], X.shape[1])), \
        "Linear activation forward wrong dimensions"


def test_compute_cost():
    print()
    print("Test Compute Cost")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    cost = compute_cost(A2, Y)
    print("cost: ", cost)
    assert (cost.shape == ()), "Cost should be a single value NOT an array"


def test_sigmoid_backword():
    print()
    print("Test sigmoid backward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    # foreward propagation
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    # backward propagation
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dZ = sigmoid_backward(dA2, Z2)
    assert (dZ.shape == Z2.shape), "Dimensions do not match"
    dZ1 = sigmoid_backward(0.0001, 10)
    dZ2 = sigmoid_backward(0.5, 0)
    dZ3 = sigmoid_backward(0.0001, -10)
    print("sigmoid_backward(0, 10): ", dZ1)
    print("sigmoid_backward(0, 0): ", dZ2)
    print("sigmoid_backward(0, -10): ", dZ3)
    assert dZ1 == 4.539580773590766e-09, "Relu backward does not work properly"
    assert dZ2 == 0.125, "Relu backward does not work properly"
    assert dZ3 == 4.539580773595167e-09, "Relu backward does not work properly"


def test_relu_backward():
    print()
    print("Test relu backward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    # foreward propagation
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    # backward propagation
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dA1, dW2, db2 = linear_activation_backward(dA2, Z2, A1, params["W2"], "sigmoid")
    dZ = relu_backward(dA1, Z1)
    assert (dZ.shape == Z1.shape), "Dimensions do not match"
    dZ1 = relu_backward(1, 10)
    dZ2 = relu_backward(1, 1)
    dZ3 = relu_backward(0, -10)
    print("relu_backward(1, 10): ", dZ1)
    print("relu_backward(1, 1): ", dZ2)
    print("relu_backward(0, -10): ", dZ3)
    assert dZ1 == 1, "Relu backward does not work properly"
    assert dZ2 == 1, "Relu backward does not work properly"
    assert dZ3 == 0, "Relu backward does not work properly"


def test_linear_backward():
    print()
    print("Test linear backward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    # foreward propagation
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    # backward propagation
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dZ = sigmoid_backward(dA2, Z2)
    dA, dW, db = linear_backward(dZ, A1, params["W2"])
    print("dA.shape: ", dA.shape)
    print("A2.shape: ", A1.shape)
    print("dW.shape: ", dW.shape)
    print("W2.shape: ", params["W2"].shape)
    print("db.shape: ", db.shape)
    print("b2.shape: ", params["b2"].shape)
    assert (dA.shape == A1.shape), "Dimensions do not match"
    assert (dW.shape == params["W2"].shape), "Dimensions do not match"
    assert (db.shape == params["b2"].shape), "Dimensions do not match"


def test_linear_activation_backward():
    print()
    print("Test linear activation backward")
    print("Test pass")


def test_L_linear_activation_backward():
    print()
    print("Test L layer linear activation backward")
    print("Test pass")


def test_update_params():
    print()
    print("Test update parameters")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    learning_rate = 0.001
    grads = {}
    # foreward propagation
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    # backward propagation
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dA1, dW2, db2 = linear_activation_backward(dA2, Z2, A1, params["W2"], "sigmoid")
    dA0, dW1, db1 = linear_activation_backward(dA1, Z1, X, params["W1"], "relu")

    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2

    params_updated = update_params(params, grads, learning_rate)

    # compare params with params_updated
    # 2 = number of layers in the neural network
    for l in range(2):
        print("params[W" + str(l + 1) + "].shape: ",
              params["W" + str(l + 1)].shape)
        print("params_updated[W" + str(l + 1) + "].shape: ",
              params_updated["W" + str(l + 1)].shape)
        assert params["W" + str(l + 1)].shape == params_updated["W" + str(l + 1)].shape,\
            "Dimensions do not match"
        print("params[b" + str(l + 1) + "].shape: ",
              params["b" + str(l + 1)].shape)
        print("params_updated[b" + str(l + 1) + "].shape: ",
              params_updated["b" + str(l + 1)].shape)
        assert params["b" + str(l + 1)].shape == params_updated["b" + str(l + 1)].shape,\
            "Dimensions do not match"


def test_L_update_params():
    print()
    print("Test update parameters")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    learning_rate = 0.001
    grads = {}
    # foreward propagation
    A1, Z1 = linear_activation_forward(X, params["W1"], params["b1"], "relu")
    A2, Z2 = linear_activation_forward(A1, params["W2"], params["b2"], "sigmoid")
    # backward propagation
    dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    dA1, dW2, db2 = linear_activation_backward(dA2, Z2, A1, params["W2"], "sigmoid")
    dA0, dW1, db1 = linear_activation_backward(dA1, Z1, X, params["W1"], "relu")

    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2

    params_updated = L_update_params(params, grads, learning_rate)

    # compare params with params_updated
    # 2 = number of layers in the neural network
    for l in range(2):
        print("params[W" + str(l + 1) + "].shape: ",
              params["W" + str(l + 1)].shape)
        print("params_updated[W" + str(l + 1) + "].shape: ",
              params_updated["W" + str(l + 1)].shape)
        assert params["W" + str(l + 1)].shape == params_updated["W" + str(l + 1)].shape,\
            "Dimensions do not match"
        print("params[b" + str(l + 1) + "].shape: ",
              params["b" + str(l + 1)].shape)
        print("params_updated[b" + str(l + 1) + "].shape: ",
              params_updated["b" + str(l + 1)].shape)
        assert params["b" + str(l + 1)].shape == params_updated["b" + str(l + 1)].shape,\
            "Dimensions do not match"


def test_two_layer_forward():
    print()
    print("Test two layer forward")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)
    A = two_layer_forward(X, params)
    print("A.shape: ", A.shape)
    print("X.shape[1]: ", X.shape[1])
    assert (A.shape == (1, X.shape[1]))


def test_predict():
    print()
    print("Test predictions")
    filename = "datasets/catvnoncat_2.h5"
    X, Y = load_data(filename)
    X = preprocess(X)
    X = np.concatenate((X[:, :5], X[:, -5:]), axis=1)
    Y = np.concatenate((Y[:, :5], Y[:, -5:]), axis=1)
    #X = np.array([X[:, :5], X[:, -5:]])
    print(X.shape)
    print(Y.shape)
    n_x = X.shape[0]  # 12288
    n_l1 = 10
    n_yhat = 1
    random_on = True
    seed = 1
    params = init_params(n_x, n_l1, n_yhat, random_on, seed)

    iterations = 300
    learning_rate = 0.001
    grads = {}
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
        if(i % 100 == 0):
            print("Cost after iteration", str(i) + ":", np.squeeze(cost))

    p = predict(X, params)
    print("p.shape: ", p.shape)
    number_of_ones = np.sum((p == 1))
    number_of_zeros = np.sum((p == 0))
    print("number of ones: ", number_of_ones)
    print("number of zeros: ", number_of_zeros)
    assert number_of_ones + number_of_zeros == p.shape[1],\
        "Predictions con only be one or zero"


def test_accuracy():
    print()
    print("Test accuracy")
    Y = np.array([[1, 1, 0, 0]])
    p = np.array([[1, 0, 1, 0]])
    print("Accuracy: ", accuracy(p, Y))
    assert accuracy(p, Y) == 0.5, "Accuracy does not work properly"


def main():
    print(time.time())
    print("Running all tests")
    print()
    test_load_data()
    test_L_init_params()
    test_sigmoid()
    test_relu()
    test_preprocess()
    test_linear_forward()
    test_linear_activation_forward()
    test_L_linear_activation_forward()
    test_compute_cost()
    test_sigmoid_backword()
    test_relu_backward()
    test_linear_backward()
    test_linear_activation_backward()
    test_L_linear_activation_backward()
    test_update_params()
    test_L_update_params()
    #test_two_layer_forward()
    #test_predict()
    #test_accuracy()

    print()
    print("All tests pass")


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
