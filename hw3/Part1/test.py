import numpy as np
import os
import matplotlib.pyplot as plt
from CE40719.module import *
from CE40719.batch_norm import *
from CE40719.cnn_batch_norm import *
from CE40719.convolution import *
from CE40719.group_batch_norm import *
from CE40719.max_pool import *


"""""
np.random.seed(40959)
x = np.random.randn(2, 3, 4, 4)
w = np.random.randn(3, 3, 4, 4)
b = np.random.randn(3, )
conv = Convolution('test', (3, 3, 4, 4), stride=2, pad=1)
conv.W = w
conv.b = b
output = conv.forward(x)
correct_output = np.array([[[[2.90745973, -1.93569447],
                             [10.18479326, 0.95405206]],
                            [[2.67941204, -1.06629218],
                             [-2.6427213, -3.12561258]],
                            [[-5.32408346, 1.12473438],
                             [4.16451343, -5.04230883]]],
                           [[[0.18517581, 10.22485798],
                             [-3.51174763, 1.9202936]],
                            [[-2.56595929, -3.40545467],
                             [0.33082083, 4.34434771]],
                            [[-3.54337648, 2.44988087],
                             [-3.6369818, 1.96857427]]]])

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

conv = Convolution('test', (2, 3, 2, 2), stride=2, pad=1)
x = np.random.randn(2, 3, 2, 2)
W = np.random.randn(2, 3, 2, 2)
b = np.random.randn(2, )

conv.W = W
conv.b = b
dout = np.random.randn(2, 2, 2, 2)
out = conv.forward(x)
dx = conv.backward(dout)

correct_dx = np.array([[[[-0.03022149, -0.93652977], [-0.05179407, 1.62558139]],
                        [[1.62356625, 3.17432728], [1.37585703, 0.21801443]],
                        [[-1.14110006, -3.2751212], [0.98650008, 0.78396852]]],
                       [[[0.48556001, 1.24240355], [0.1635526, 0.97860699]],
                        [[2.07933521, -1.62650629], [-0.35726596, 0.17660094]],
                        [[0.27806844, 2.30231871], [-1.07156607, 0.22142858]]]])
#print(dx - correct_dx)
print('Relative error dx:', np.linalg.norm(dx - correct_dx))

correct_dW = np.array([[[[0.16162545, 0.44372442], [0.5131281, 0.41785749]],
                        [[-0.4409529, -0.31301584], [-0.18734267, 0.06869406]],
                        [[-0.53426167, -0.94735183], [0.9614619, 0.36417281]]],
                       [[[-0.40656537, -0.1906337], [1.38892306, -0.59866861]],
                        [[0.81392044, 0.36665929], [0.78840142, 2.80736748]],
                        [[1.58139656, -0.81670389], [-1.11075549, -1.7656368]]]])
print('Relative error dw:', np.linalg.norm(conv.dW - correct_dW))

correct_db = np.array([0.82375129, 2.84032899])
print('Relative error db:', np.linalg.norm(conv.db - correct_db))


np.random.seed(40959)
x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool = MaxPool('test',height = 2,width = 2, stride = 2)
output = pool.forward(x)

correct_output = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

x = np.random.randn(3, 2, 2, 2)
dout = np.random.randn(3, 2, 1, 1)
pool = MaxPool('test',height = 2, width = 2, stride =2)
out = pool.forward(x)
dx = pool.backward(dout)
correct_dx = np.array([[[[0., 1.21236066],[0., 0.]],
                        [[0.45107133, 0.],[0., 0.]]],
                       [[[0., 0.],[-0.86463156,0.]],
                        [[0., 0.],[0., -0.39180953]]],
                       [[[0.93694169, 0.],[0., 0.]],
                        [[0., 0.],[-0.08002411, 0.]]]])
print('Relative error dx:', np.linalg.norm(dx - correct_dx))

"""""

###########################################################################
#            Batch-Normalization forward pass Test                        #
###########################################################################
np.random.seed(40959)
# Check the training-time forward pass by checking means and variances
# of features both before and after spatial batch normalization

N, C, H, W = 2, 3, 4, 5
x = 4 * np.random.randn(N, C, H, W) + 10

print('Before spatial batch normalization:')
print('  Shape: ', x.shape)
print('  Means: ', x.mean(axis=(0, 2, 3)))
print('  Stds: ', x.std(axis=(0, 2, 3)))

# Means should be close to zero and stds close to one
gamma, beta = np.ones(C), np.zeros(C)
cnn_batchnorm = CnnBatchNorm('test',(N, C, H, W))
cnn_batchnorm.batchnorm.gamma = gamma
cnn_batchnorm.batchnorm.beta = beta
out = cnn_batchnorm.forward(x)
print('After spatial batch normalization:')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(axis=(0, 2, 3)))
print('  Stds: ', out.std(axis=(0, 2, 3)))

# Means should be close to beta and stds close to gamma
gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
cnn_batchnorm.batchnorm.gamma = gamma
cnn_batchnorm.batchnorm.beta = beta
out= cnn_batchnorm.forward(x)
print('After spatial batch normalization (nontrivial gamma, beta):')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(axis=(0, 2, 3)))
print('  Stds: ', out.std(axis=(0, 2, 3)))

np.random.seed(40959)
# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.
N, C, H, W = 10, 4, 11, 12
cnn_batchnorm = CnnBatchNorm('test',(N, C, H, W))
cnn_batchnorm.train()
for t in range(50):
  x = 2.3 * np.random.randn(N, C, H, W) + 13
  cnn_batchnorm.forward(x)
cnn_batchnorm.test()
x = 2.3 * np.random.randn(N, C, H, W) + 13
a_norm = cnn_batchnorm.forward(x)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print('After spatial batch normalization (test-time):')
print('  means: ', a_norm.mean(axis=(0, 2, 3)))
print('  stds: ', a_norm.std(axis=(0, 2, 3)))

###########################################################################
#            Batch-Normalization backward pass Test                       #
###########################################################################
np.random.seed(40959)
N, C, H, W = 2, 3, 2, 2
x = 5 * np.random.randn(N, C, H, W) + 12
gamma = np.random.randn(C)
beta = np.random.randn(C)
dout = np.random.randn(N, C, H, W)
cnn_batchnorm = CnnBatchNorm('Test',(N, C, H, W))
cnn_batchnorm.batchnorm.gamma = gamma
cnn_batchnorm.batchnorm.beta = beta
cnn_batchnorm.train()
_ = cnn_batchnorm.forward(x)
dx = cnn_batchnorm.backward(dout)
dgamma = cnn_batchnorm.dgamma
dbeta = cnn_batchnorm.dbeta
correct_dx = np.array([[[[0.00589789,  1.2557341 ],[-0.18515455, -0.3084614 ]],
                        [[-0.04023214, -0.11912787],[-0.04556006, -0.00270806]],
                        [[ 0.12266522, -0.07093585],[ 0.22957267,  0.17611092]]],
                       [[[ 0.36047414, -0.01314037],[-0.62981818, -0.48553163]],
                        [[ 0.18630326, -0.02134853],[-0.15169621,  0.19436962]],
                        [[-0.00739465, -0.04518148],[-0.2105455,  -0.19429132]]]])
print('Relative error dx:', np.linalg.norm(dx - correct_dx))
correct_dgamma = np.array([ 1.51945006, -1.09337409,  0.8928227 ])
print('Relative error dgamma:', np.linalg.norm(dgamma - correct_dgamma))
correct_dbeta = np.array([-3.1690584,   3.01154949,  5.44132887])
print('Relative error dbeta:', np.linalg.norm(dbeta - correct_dbeta))