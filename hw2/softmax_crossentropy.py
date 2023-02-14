import numpy as np
from module import Module


class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        **Save whatever you need for backward pass in self.cache.
        """
        N = x.shape[0]
        x = (x.T - np.max(x, axis=1)).T
        exp_x = np.exp(x)
        normalization = np.sum(exp_x, axis=1)
        probs = (exp_x.T / normalization).T
        loss = -np.sum(np.log(probs[range(N), y]))/N
        grad = np.copy(probs)
        grad[range(N), y] -= 1
        self.cache = grad/N
        # todo: implement the forward propagation for probs and compute cross entropy loss
        # NOTE: implement a numerically stable version.If you are not careful here
        # it is easy to run into numeric instability!
        return loss, probs

    def backward(self, dout=0):
        dx = self.cache
        # todo: implement the backward propagation for this layer.

        return dx
