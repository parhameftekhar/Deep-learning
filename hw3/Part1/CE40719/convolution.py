import numpy as np
from CE40719.module import Module


class Convolution(Module):
    def __init__(self, name, filter_shape, stride = 1, pad =0, weight_scale =1e-3, l2_coef=.0):
        super(Convolution, self).__init__(name)
        self.W = np.random.normal(scale = weight_scale, size = filter_shape)  # filter of the layer with shape (F, C, f_h, f_w).
        self.b = np.zeros(filter_shape[0], )  # biases of the layer with shape (F,).
        self.dW = None  # gradients of loss w.r.t. the weights.
        self.db = None  # gradients of loss w.r.t. the biases.
        self.stride = stride
        self.pad = pad
        self.l2_coef = l2_coef

    def forward(self, x, **kwargs):
        """
        x: input array.
        out: output of convolution module for input x.
        Save whatever you need for backward pass in self.cache.
        """
        out = None
        # todo: implement the forward propagation for Dense module.
        N, C, H, W = x.shape
        F, _, f_h, f_w = self.W.shape
        pad = self.pad
        stride = self.stride
        out_h = (H + 2 * pad - f_h) // stride + 1
        out_w = (W + 2 * pad - f_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, f_h, f_w, out_h, out_w))
        #col1 = np.zeros((N, C, f_h, f_w, out_h, out_w))
        for y in range(f_h):
            y_max = y + stride * out_h
            for x in range(f_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = \
                    img[:, :, y:y_max:stride, x:x_max:stride]

       #for y in range(out_h):
       #     for x in range(out_w):
       #         col1[:, :, :, :, y, x] = img[:, :, y*stride:y*stride+f_h, x*stride:x*stride+f_w]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
       # col1 = col1.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        col_W = self.W.reshape(F, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.cache = ((N, C, H, W), col, col_W)
        return out

    def backward(self, dout):
        """
        dout: gradients of Loss w.r.t. this layer's output.
        dx: gradients of Loss w.r.t. this layer's input.
        """
        dx = None
        def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
        
            N, C, H, W = input_shape
            out_h = (H + 2 * pad - filter_h) // stride + 1
            out_w = (W + 2 * pad - filter_w) // stride + 1
            col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

            img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
            for y in range(filter_h):
                y_max = y + stride * out_h
                for x in range(filter_w):
                    x_max = x + stride * out_w
                    img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

            return img[:, :, pad:H + pad, pad:W + pad]
        FN, C, FH, FW = self.W.shape
        N, FN, H, W = dout.shape
        #dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        dout = dout.transpose(1, 0, 2, 3)
        dout = dout.reshape(FN, -1)
        x_shape, col, col_W = self.cache
        self.db = np.sum(dout.T, axis=0)
        self.dW = np.dot(col.T, dout.T)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout.T, col_W.T)
        dx = col2im(dcol, x_shape, FH, FW, self.stride, self.pad)

        # todo: implement the backward propagation for Dense module.
        # don't forget to update self.dW and self.db.
        
        return dx

