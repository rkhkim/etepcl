from __future__ import absolute_import, division, print_function
from scipy.optimize import minimize
import numpy as np
import csv
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from skimage import io
import pylab as plot
import keras
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from math import ceil



# supress numpy error warnings so as not to hinder the cli output
np.seterr(all='ignore')

class Conv():
    def network(activation_function='relu'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation(activation_function))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(activation_function))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(activation_function))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

    def cnn(model, epochs=1, opt = keras.optimizers.SGD(lr=0.01, momentum=0.7, decay=0.001), verbose=2):
        num_classes = 10

        # The data, split between train, validation, and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        (x_train, x_validate, y_train, y_validate) = train_test_split(x_train, y_train, test_size=0.2)
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'training samples')
        print(x_validate.shape[0], 'validation samples')
        print(x_test.shape[0], 'test samples')
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_validate = keras.utils.to_categorical(y_validate, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_validate = x_validate.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_validate /= 255
        x_test /= 255


      batch_size = 32
      save_dir = os.path.join(os.getcwd(), 'saved_models')
      model_name = 'keras_cifar10_trained_model.h5'

      # Configure the model for training
      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

      history = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_validate, y_validate),
                          shuffle=True,
                          verbose=verbose)

      # Save model and weights
      if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
      model_path = os.path.join(save_dir, model_name)
      model.save(model_path)

      # Score trained model.
      scores = model.evaluate(x_test, y_test, verbose=1)
      print('Test loss:', scores[0])
      print('Test accuracy:', scores[1])
      return [history.epoch, history.history['acc'], history.history['val_acc']]

class NeuralNet():
    """The main neural network class for training"""

    def PCA(X, varRetained = 0.95, show = False):

        # Compute Covariance Matrix Sigma
        (n, m) = X.shape
        Sigma = 1.0 / m * X * np.transpose(X)
        # Compute eigenvectors and eigenvalues of Sigma
        U, s, V = np.linalg.svd(Sigma, full_matrices = True)

        # compute the value k: number of minumum features that
        # retains the given variance
        sTot = np.sum(s)
        var_i = np.array([np.sum(s[: i + 1]) / \
                            sTot * 100.0 for i in range(n)])
        k = len(var_i[var_i < (varRetained * 100)])


        # plot the variance plot
        plot.plot(var_i)
        plot.xlabel('Number of Features')
        plot.ylabel(' Percentage Variance retained')
        plot.title('PCA $\% \sigma^2 $ vs # features')
        plot.show()

        # compute the reduced dimensional features by projction
        U_reduced = U[:, : k]
        Z = np.dot(np.transpose(U_reduced),X)

        return Z, U_reduced


    def bandpass(img):
        img = np.ndarray.astype(img, 'float')
        sz = img.shape
        depth = np.rint(np.log2(sz))[0].astype(int)
        assert np.all(2 ** depth == sz[:2]), 'Image must be square with 2^k rows'
        E = np.array([[1, -1, 1, -1], [1, 1, -1, -1],
                      [1, -1, -1, 1], [1, 1, 1, 1]]) / 4;
        detail = []
        for level in range(depth):
            Q = [img[::2, ::2], img[1::2, ::2], img[::2, 1::2], img[1::2, 1::2]]
            det = []
            for d in range(4):
                det.append(np.zeros_like(Q[0]))
                for q in range(4):
                    det[d] += E[d, q] * Q[q]
            detail.append(det[:3])
            img = det[3]
        return detail, img

    def composite(detail, mean):
        n = detail[0][0].shape[0]
        n2 = n * 2
        comp = np.full((n2, n2), mean[0, 0])
        a, b, c = 0, n, n2
        for d in detail:
            comp[a:b, b:c] = d[0]
            comp[b:c, a:b] = d[1]
            comp[a:b, a:b] = d[2]
            a, n = b, int(n/2)
            b += n
        return comp

    def show(img, size=(5, 5), cmap='gray'):
        plt.figure(figsize=size)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()

    def __init__(self, X, Y, writer, output="./params", lam=1, maxiter=250, norm=False):
        """
        Arguments:
            X {np.ndarray} -- The training set
            Y {np.ndarray} -- The expected output of the training set
            writer {class} -- A writer interface which implements the write() method

        Keyword Arguments:
            output {str} -- Where to save the trained params to (default: {"./params"})
            lam {number} -- Lambda term for regularization (default: {1})
            maxiter {number} -- Max iterations for minimization (default: {250})
        """

        X = np.matrix(X)
        Y = np.matrix(Y)
        if norm:
            X = normalize(X)

        self.X = X
        self.Y = Y

        self.num_labels = np.shape(Y)[1]
        self.input_size = np.shape(X)[1]
        self.hidden_size = np.shape(X)[1]

        self.lam = lam
        self.output = output
        self.params = self.generate_params()
        self.maxiter = maxiter
        self.writer = writer
        self.minval = 0.000000001

    def set_input_size(self, input_size):
        """
        set the input size of the network

        Arguments:
            input_size {int}
        """

        self.input_size = input_size

    def set_num_labels(self, num_labels):
        """
        set the num of labels of output

        Arguments:
            num_labels {int}
        """

        self.num_labels = num_labels

    def set_hidden_size(self, hidden_size):
        """
        set the hidden layer size

        Arguments:
            hidden_size {int}
        """

        self.hidden_size = hidden_size
    def set_params(self, params):
        """
        set the params

        Arguments:
            params {np.ndarray} -- params
        """

        self.params = params

    def set_X(self, X, norm=True):
        """
        Set the training set

        Arguments:
            X {np.ndarray} -- The input layer

        Keyword Arguments:
            norm {bool} -- Should the input be normalized (default: {True})
        """

        if norm:
            X = normalize(X)

        self.X = X

    def set_Y(self, Y):
        """
        Set the expected output

        Arguments:
            Y {np.ndarray} -- The expected output
        """

        self.Y = Y

    def train(self, verbose=False, save=True):
        """
        minimize a cost function defined under backpropogation

        Keyword Arguments:
            verbose {bool} -- should the backpropgation print progress (default: {False})
            save {bool} -- should output of parameters be saved to a file (default: {True})

        Returns:
            np.ndarray
        """

        fmin = minimize(fun=self.fit, x0=self.params, args=(self.X, self.Y, verbose),
                        method='TNC', jac=True, options={'maxiter': self.maxiter})

        if save:
            writer = csv.writer(open(self.output, 'w'))
            writer.writerow(fmin.x)

        self.params = fmin.x

    def generate_params(self):
        """
        generate a random sequence of weights for the
        parameters of the neural network

        Returns:
            np.ndarray
        """

        return (np.random.random(size=self.hidden_size * (self.input_size + 1) + self.num_labels * (self.hidden_size + 1)) - 0.5) * 0.25

    def load_params(self, name):
        """
        load parameters from a csv file

        Arguments:
            name {string} -- the location of the file

        Returns:
            np.ndarray -- the loaded params
        """

        return np.loadtxt(open(name,"rb"), delimiter=",",skiprows=0, dtype="float")

    def sigmoid(self, z):
        """
        compute the sigmoid activation function

        Arguments:
            z {mixed}

        Returns:
            number
        """

        return 1 / (1 + np.exp(-z))

    def sigmoid_gradient(self, z):
        """
        gradient of the sigmoid func

        Arguments:
            z {mixed}

        Returns:
            np.ndarray|float
        """

        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def reshape_theta(self, params):
        """
        reshape the 1 * n parameter vector into the correct shape for the first and second layers

        Arguments:
            params {np.ndarray} -- a vector of weights

        Returns:
            theta1 {np.ndarray}
            theta2 {np.ndarray}
        """
        theta1 = np.matrix(np.reshape(params[:self.hidden_size * (self.input_size + 1)], (self.hidden_size, (self.input_size + 1))))
        theta2 = np.matrix(np.reshape(params[self.hidden_size * (self.input_size + 1):], (self.num_labels, (self.hidden_size + 1))))

        return theta1, theta2

    def feed_forward(self, X, theta1, theta2):
        """
        run forward propgation using a value of X

        Arguments:
            X {np.ndarray} -- Input set
            theta1 {np.ndarray} -- The first layer weights
            theta2 {np.ndarray} -- The second layer weights

        Returns:
            a1 {np.ndarray}
            z2 {np.ndarray}
            a2 {np.ndarray}
            z3 {np.ndarray}
            h  {np.ndarray}
        """

        m = X.shape[0]

        a1 = np.insert(X, 0, values=np.ones(m), axis=1)

        z2 = a1 * theta1.T
        a2 = np.insert(self.sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = self.sigmoid(z3)

        return a1, z2, a2, z3, h

    def fit(self, params, X, y, output=True):
        """
        main function to run a single pass on the nn. First run forward propgation to get the error of output given some
        parameters and then perfom backpropgation to work out the gradient of the function using the given weights.

        Arguments:
            params {np.ndarray} -- weight layer parameters
            X {np.ndarray} -- Input matrix
            y {np.ndarray} -- Expected output matrix

        Keyword Arguments:
            output {bool} -- print to the writer (default: {True})

        Returns:
            J {float64} -- the margin of error with the given weights
            grad {np.ndarray} -- the matrix of gradients for the given weights
        """

        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)

        theta1, theta2 = self.reshape_theta(params)
        a1, z2, a2, z3, h = self.feed_forward(X, theta1, theta2)

        # initializations
        J = 0
        delta1 = np.zeros(theta1.shape)
        delta2 = np.zeros(theta2.shape)

        J = self.get_cost(y, h) / m

        J += (float(self.lam) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
        if output:
            self.writer.write(J)

        for t in range(m):
            a1t = a1[t,:]
            z2t = z2[t,:]
            a2t = a2[t,:]
            ht = h[t,:]
            yt = y[t,:]

            d3t = ht - yt

            z2t = np.insert(z2t, 0, values=np.ones(1))
            d2t = np.multiply((theta2.T * d3t.T).T, self.sigmoid_gradient(z2t))

            delta1 = delta1 + (d2t[:,1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t

        delta1 = delta1 / m
        delta2 = delta2 / m

        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * self.lam) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * self.lam) / m

        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

        return J, grad

    def get_cost(self, y, h):
        """
        get the cost of prediction, the error margin

        Arguments:
            y {np.ndarray} -- The expected output
            h {np.ndarray} -- The prediction array

        Returns:
            cos {float64} -- the margin of error with the given weights
        """
        first_term = np.multiply(-y, np.log(h.clip(self.minval)))
        second_term = np.multiply((1 - y), np.log(1 - h.clip(self.minval)))

        return np.sum(first_term - second_term)

    def accuracy(self, type='train'):
        """
        get the accuracy of the learned parameters on a specific set
        """

        examples = len(self.Y)
        theta1, theta2 = self.reshape_theta(self.params)

        a1, z2, a2, z3, h = self.feed_forward(self.X, theta1, theta2)
        y_pred = np.array(np.argmax(h, axis=1))

        correct = 0
        for x in range(examples):
            if self.Y[x, y_pred[x]] == 1:
                correct += 1

        accuracy = (correct / float(examples))
        self.writer.write(type +' accuracy = {0}%'.format(accuracy * 100))

    def predict(self, x):
        """
        predict given a row example

        Arguments:
            x {np.array} -- the feature row used to predict and output

        Returns:
            np.array -- the prediction
        """

        theta1, theta2 = self.reshape_theta(self.params)
        _,_,_,_,h = self.feed_forward(x, theta1, theta2)
        return np.array(np.argmax(h, axis=1))[0,0]

    def test(self, step=10):
        """
        run a diagnostic check on the given data set and expected output. This method plots the the margin of prediction
        error against the increase in size of training examples. This can be useful to determine what is going wrong
        with your hypothesis, i.e. whether it is underfitting or overfitting the training set.

        Arguments:
            X {[type]} -- The input set
            Y {[type]} -- The expected output

        Keyword Arguments:
            step {number} -- The size of step taken in to increase the dataset (default: {10})
        """
        # split into 6/2/2 ratio train/cv/test
        x_train, x_cross_validation, x_test = split(self.X)
        y_train, y_cross_validation, y_test = split(self.Y)

        error_train = []
        error_val = []
        amount = 0
        i = 1
        while i < len(x_train):
            self.writer.write("running at index %s of %s" % (i, len(x_train)))
            params = self.generate_params()
            current_input = x_train[0:i, :]
            current_output = y_train[0:i, :]

            fmin = minimize(fun=self.fit, x0=params, args=(current_input, current_output, False),
                            method='TNC', jac=True, options={'maxiter': self.maxiter})
            train_cost, _= self.fit(fmin.x, current_input, current_output, False)
            val_cost, _ = self.fit(fmin.x, x_cross_validation, y_cross_validation, False)

            error_train.append(train_cost)
            error_val.append(val_cost)

            amount += 1
            i = amount * step

        plt.plot(error_train)
        plt.plot(error_val)

        plt.legend(['train', 'validation'], loc='upper left')
        plt.ylabel("error")

        plt.xlabel("Iteration")
        plt.show()

def split(input):
    """[summary]

    [description]

    Arguments:
        input {np.ndarray} -- The input set

    Returns:
        train_set {np.ndarray}
        cross_set {np.ndarray}
        test_set {np.ndarray}
    """

    length = len(input)
    unit = length/10

    train = int(round(unit*6, 0))
    cross_test = int(round(unit*2, 0))

    train_set = input[0:train, :]
    cross_set = input[train:train+cross_test, :]
    test_set = input[train+cross_test: length, :]

    return train_set, cross_set, test_set
