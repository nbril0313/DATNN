import os
import time
import numpy as np
from math import sqrt


class DANN(object):

    def __init__(self, learning_rate=0.05, hidden_layer_size=25, lambda_adapt=1., maxiter=200,
                 epsilon_init=None, adversarial_representation=True, seed=12342, verbose=False):
        """
        Domain Adversarial Neural Network for classification

        option "learning_rate" is the learning rate of the neural network.
        option "hidden_layer_size" is the hidden layer size.
        option "lambda_adapt" weights the domain adaptation regularization term.
                if 0 or None or False, then no domain adaptation regularization is performed
        option "maxiter" number of training iterations.
        option "epsilon_init" is a term used for initialization.
                if None the weight matrices are weighted by 6/(sqrt(r+c))
                (where r and c are the dimensions of the weight matrix)
        option "adversarial_representation": if False, the adversarial classifier is trained
                but has no impact on the hidden layer representation. The label predictor is
                then the same as a standard neural-network one (see experiments_moon.py figures).
        option "seed" is the seed of the random number generator.
        """

        self.hidden_layer_size = hidden_layer_size
        self.maxiter = maxiter
        self.lambda_adapt = lambda_adapt if lambda_adapt not in (None, False) else 0.
        self.epsilon_init = epsilon_init
        self.learning_rate = learning_rate
        self.adversarial_representation = adversarial_representation
        self.seed = seed
        self.verbose = verbose

    def sigmoid(self, z):
        """
        Softmax function.

        """
        v = np.exp(z)
        return v / np.sum(v, axis=0)

    def softmax(self, z):
        """
        Softmax function.

        """
        v = np.exp(z)
        return v / np.sum(v, axis=0)

    def random_init(self, l_in, l_out):
        """
        This method is used to initialize the weight matrices of the DA neural network

        """
        if self.epsilon_init is not None:
            epsilon = self.epsilon_init
        else:
            epsilon = sqrt(6.0 / (l_in + l_out))

        return epsilon * (2 * np.random.rand(l_out, l_in) - 1.0)

    def fit(self, X, Y, X_adapt, X_valid=None, Y_valid=None, do_random_init=True):
        """
        Trains the domain adversarial neural network until it reaches a total number of
        iterations of "self.maxiter" since it was initialize.
        inputs:
              X : Source data matrix
              Y : Source labels
              X_adapt : Target data matrix
              (X_valid, Y_valid) : validation set used for early stopping.
              do_random_init : A boolean indicating whether to use random initialization or not.
        """
        nb_examples, nb_features = np.shape(X)
        nb_labels = len(set(Y))
        nb_examples_adapt, _ = np.shape(X_adapt)

        if self.verbose:
            print('[DANN parameters]', self.__dict__)

        np.random.seed(self.seed)

        if do_random_init:
            W = self.random_init(nb_features, self.hidden_layer_size)
            V =