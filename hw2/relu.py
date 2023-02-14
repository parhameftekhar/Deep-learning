import numpy as np
from module import Module


class ReLU(Module):
    def __init__(self, name):
        super(ReLU, self).__init__(name)

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of ReLU function for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        mask = x > 0
        x[np.logical_not(mask)] = 0
        out = x
        self.cache = mask.astype(np.uint8)
        # todo: implement the forward propagation for ReLU module.

        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """

        dx = self.cache * dout
        # todo: implement the backward propagation for ReLU module.

        return dx

