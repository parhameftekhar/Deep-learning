import numpy as np
from module import Module


class Dropout(Module):
    def __init__(self, name, keep_prob=1.):
        super(Dropout, self).__init__(name)
        self.keep_prob = keep_prob  # probability of keeping neurones.

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of Drop out module for input x.
        **Save whatever you need for backward pass in self.cache.
        """
        if self.phase == 'Train':
            # todo: implement the forward propagation for Dropout module for train phase.
            N, D = x.shape
            mask = np.random.rand(N, D) < self.keep_prob
            self.cache = mask / self.keep_prob
            out = x*self.cache

        else:
            # todo: implement the forward propagation for Dropout module for test phase.
            out = x
        return out.astype(x.dtype, copy=False)

    def backward(self, dout):
        """
         dout: gradients of Loss w.r.t. this layer's output.
         dx: gradients of Loss w.r.t. this layer's input.
         """
        # todo: implement the backward propagation for Dropout module.(train phase only)
        dx = dout * self.cache

        return dx
