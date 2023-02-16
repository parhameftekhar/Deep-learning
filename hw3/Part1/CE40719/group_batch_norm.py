import numpy as np
from CE40719.module import Module


class GroupBatchNorm(Module):
    def __init__(self, name, input_shape, G =1, epsilon=1e-5):
        super(GroupBatchNorm, self).__init__(name)
        N, C, H, W = input_shape
        self.gamma = np.ones((1,C,1,1)) # Scale parameter, of shape (C,)
        self.beta = np.zeros((1,C,1,1)) # Shift parameter, of shape (C,)
        self.G = G # Integer number of groups to split into, should be a divisor of C
        self.eps = epsilon
        self.dbeta = 0
        self.dgamma = 0
    def forward(self, x, **kwargs):
        """
        Computes the forward pass for spatial group normalization.
        In contrast to layer normalization, group normalization splits each entry 
        in the data into G contiguous pieces, which it then normalizes independently.
        Per feature shifting and scaling are then applied to the data, in a manner 
        identical to that of batch normalization.
        **Save whatever you need for backward pass in self.cache.
        """
        out = None
        # TODO: Implement the forward pass for spatial group normalization.
        
        return out

    def backward(self, dout):
        """
        Computes the backward pass for spatial group normalization.
        dx: Gradient with respect to inputs, of shape (N, C, H, W)
        dgamma: Gradient with respect to scale parameter, of shape (C,)
        dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx = None
        # TODO: Implement the backward pass for spatial group normalization.
        # don't forget to update self.dgamma and self.dbeta.
        

        return dx

