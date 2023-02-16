import numpy as np
from CE40719.module import Module


class MaxPool(Module):
    def __init__(self, name,height=1,width=1,stride=1):
        super(MaxPool, self).__init__(name)
        self.height=height #The height of each pooling region
        self.width=width   #The width of each pooling region
        self.stride=stride #The distance between adjacent pooling regions

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of max pool module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        F_H, F_W = self.height, self.width
        s = self.stride
        N, C, I_h, I_w = x.shape
        O_H = (I_h - F_H)//s + 1
        O_W = (I_w - F_W)//s + 1
        out = np.zeros((N, C, O_H, O_W))
        indices = np.zeros((N, C, O_H, O_W, F_H, F_W))
        for oh in range(O_H):
            for ow in range(O_W):
                out[:, :, oh, ow] = np.amax(x[:, :, oh*s:oh*s+F_H, ow*s:ow*s+F_W], (2,3))
                indices[:, :, oh, ow, :, :] = \
                    (out[:, :, oh, ow][:,:,np.newaxis, np.newaxis] == x[:, :, oh*s:oh*s+F_H, ow*s:ow*s+F_W]).astype(np.uint8)
        # todo: implement the forward propagation for max_pool module.
        self.cache = (indices, x.shape)
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        indices, input_shape = self.cache
        N, C, I_h, I_w = input_shape
        _, _, O_H, O_W, F_H, F_W = indices.shape
        s = self.stride
        dx = np.zeros(input_shape)
        for oh in range(O_H):
            for ow in range(O_W): 
                dx[:,:,oh*s:oh*s+F_H, ow*s:ow*s+F_W] += \
                    indices[:, :, oh, ow, :, :] * dout[:, :, oh, ow][:,:,np.newaxis,np.newaxis]
        # todo: implement the backward propagation for Dense module.
        
        return dx

