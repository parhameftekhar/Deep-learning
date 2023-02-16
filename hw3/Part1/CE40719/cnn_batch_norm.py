import numpy as np
from CE40719.module import Module
from CE40719.batch_norm import BatchNormalization

class CnnBatchNorm(Module):
    def __init__(self, name, input_shape, momentum=.9, epsilon=1e-5):
        super(CnnBatchNorm, self).__init__(name)
        N, C, H, W = input_shape
        self.momentum = momentum  # momentum rate for computing running_mean and running_var
        self.gamma = np.ones(C)  # Scale parameter, of shape (C,).
        self.beta = np.zeros(C)  # Shift parameter, of shape (C,).
        self.eps = epsilon  # this parameter will be used to avoid division by zero!

        self.dbeta = 0  # gradients of loss w.r.t. the beta parameters.
        self.dgamma = 0  # gradients of loss w.r.t. the gamma parameters.
        self.batchnorm = BatchNormalization('test', C)
        self.batchnorm.gamma = self.gamma
        self.batchnorm.beta = self.beta
    def forward(self, x, **kwargs):
        """
        x: input array of shape (N, C, H, W)
        out: output of cnn batch norm module for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        if self.phase == 'Train':
            self.batchnorm.train()
        else:
            self.batchnorm.test()
        out =  None
        # todo: implement the forward propagation for cnn batch norm module.
        # use self.batchnorm.forward()
        # Your implementation should be very short.    
        N ,C ,H ,W = x.shape 
        out = self.batchnorm.forward(x.transpose(0,2,3,1).reshape(-1, C))
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        """
        dout: input the array gradients of Loss w.r.t. this layer's output.
        dx: output, gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        # todo: implement the backward propagation for cnn batch norm module.
        # use self.batchnorm.backward()
        # don't forget to update self.dgamma and self.dbeta.
        # Your implementation should be very short.
        N ,C ,H ,W = dout.shape 
        dx = self.batchnorm.backward(dout.transpose(0,2,3,1).reshape(-1, C))
        dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        self.dgamma = self.batchnorm.dgamma
        self.dbeta = self.batchnorm.dbeta
        return dx

