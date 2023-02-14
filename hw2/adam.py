import numpy as np
from optimizer import Optimizer
from dense import Dense
from batch_norm import BatchNormalization


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-2, beta1=.9, beta2=.999, epsilon=1e-8):
        super(Adam, self).__init__(learning_rate)
        self.beta1 = beta1  # rate for Moving average of gradient.(m)
        self.beta2 = beta2  # rate for Moving average of squared gradient.(v)
        self.eps = epsilon  # this parameter will be used to avoid division by zero!
        self.v = {}  # Moving average of squared gradients.
        self.m = {}  # Moving average of gradients.
        # m and v are dictionaries that for each module save a dictionary by their names.
        # like {module_name: m_dict} or {module_name: v_dict}
        # for Dense modules v_dict and m_dict are dictionaries like {"W": v of W, "b": v of b} 
	    # or {"W": m of W, "b": m of b}
        # for Batch Norm modules v_dict and m_dict are dictionaries like {"gamma": v of gamma, "beta": v of beta}
        #       or {"gamma": m of gamma, "beta": m of beta}

    def update(self, module):
        if not (isinstance(module, Dense) or isinstance(module, BatchNormalization)):
            return  # the only modules that contain trainable parameters are dense and batch norm.

        # todo: implement adam update rules for both Dense and Batch Norm modules.
        if isinstance(module, Dense):
            m_W, m_b = self.m[module.name]['W'], self.m[module.name]['b']
            v_W, v_b = self.v[module.name]['W'], self.v[module.name]['b']

            dW = module.dW
            db = module.db

            m_W = self.beta1 * m_W + (1 - self.beta1) * dW
            m_b = self.beta1 * m_b + (1 - self.beta1) * db

            v_W = self.beta2 * v_W + (1 - self.beta2) * (dW**2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (db**2)

            self.m[module.name]['W'], self.m[module.name]['b'] = m_W, m_b
            self.v[module.name]['W'], self.v[module.name]['b'] = v_W, v_b

            t = self.iteration_number
            vt_W, vt_b = v_W / (1 - self.beta2**t), v_b / (1 - self.beta2**t)
            mt_W, mt_b = m_W / (1 - self.beta1**t), m_b / (1 - self.beta1**t)

            module.W += (-self.learning_rate / (np.sqrt(vt_W + self.eps))) * mt_W
            module.b += (-self.learning_rate / (np.sqrt(vt_b + self.eps))) * mt_b
        else:
            m_gamma, m_beta = self.m[module.name]['gamma'], self.m[module.name]['beta']
            v_gamma, v_beta = self.v[module.name]['gamma'], self.v[module.name]['beta']

            dgamma = module.dgamma
            dbeta = module.dbeta

            m_gamma = self.beta1 * m_gamma + (1 - self.beta1) * dgamma
            m_beta = self.beta1 * m_beta + (1 - self.beta1) * dbeta

            v_gamma = self.beta2 * v_gamma + (1 - self.beta2) * (dgamma ** 2)
            v_beta = self.beta2 * v_beta + (1 - self.beta2) * (dbeta ** 2)

            self.m[module.name]['gamma'], self.m[module.name]['beta'] = m_gamma, m_beta
            self.v[module.name]['gamma'], self.v[module.name]['beta'] = v_gamma, v_beta

            t = self.iteration_number
            vt_gamma, vt_beta = v_gamma / (1 - self.beta2 ** t), v_beta / (1 - self.beta2 ** t)
            mt_gamma, mt_beta = m_gamma / (1 - self.beta1 ** t), m_beta / (1 - self.beta1 ** t)

            module.gamma += (-self.learning_rate / (np.sqrt(vt_gamma + self.eps))) * mt_gamma
            module.beta += (-self.learning_rate / (np.sqrt(vt_beta + self.eps))) * mt_beta

