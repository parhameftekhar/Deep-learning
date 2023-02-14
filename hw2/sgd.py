import numpy as np
from optimizer import Optimizer
from dense import Dense
from batch_norm import BatchNormalization


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-2, momentum=.9):
        super(SGD, self).__init__(learning_rate)
        self.momentum = momentum  # momentum rate
        self.velocities = {}  # a moving average of the gradients.
        # velocities is a dictionary that saves a dictionary for each module by its name.
        # {module_name: v_dict}
        # for Dense modules v_dict is a dictionary like {"W": velocity of W, "b": velocity of b}
        # for Batch Norm modules v_dict is a dictionary like {"gamma": velocity of gamma, "beta": velocity of beta}

    def update(self, module):
        if not (isinstance(module, Dense) or isinstance(module, BatchNormalization)):
            return  # the only modules that contain trainable parameters are dense and batch norm.

        # todo: implement sgd + momentum update rules for both Dense and Batch Norm modules.
        if isinstance(module, Dense):
            W_velocity = self.velocities[module.name]['W']
            b_velocity = self.velocities[module.name]['b']
            dW = module.dW
            db = module.db
            W_velocity = self.momentum * W_velocity - self.learning_rate * dW
            b_velocity = self.momentum * b_velocity - self.learning_rate * db
            self.velocities[module.name]['W'] = W_velocity
            self.velocities[module.name]['b'] = b_velocity
            module.W += W_velocity
            module.b += b_velocity
        else:
            gamma_velocity = self.velocities[module.name]['gamma']
            beta_velocity = self.velocities[module.name]['beta']
            dgamma = module.dgamma
            dbeta = module.dbeta
            gamma_velocity = self.momentum * gamma_velocity - self.learning_rate * dgamma
            beta_velocity = self.momentum * beta_velocity - self.learning_rate * dbeta
            self.velocities[module.name]['gamma'] = gamma_velocity
            self.velocities[module.name]['beta'] = beta_velocity
            module.gamma += gamma_velocity
            module.beta += beta_velocity

