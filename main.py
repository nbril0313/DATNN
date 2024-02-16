import os
import time
import numpy as np
from math import sqrt


class DANN(object):

    def __init__(self, learning_rate=0.05, hidden_layer_size=25, lambda_adapt=1., maxiter=200,
                 epsilon_init=None, adversarial_representation=True, seed=12342, verbose=False):
