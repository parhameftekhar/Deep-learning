import numpy as np
import os
import matplotlib.pyplot as plt
from deep.HW2.model import *
from deep.HW2.adam import *
from deep.HW2.batch_norm import *
from deep.HW2.dense import *
from deep.HW2.dropout import *
from deep.HW2.module import *
from deep.HW2.optimizer import *
from deep.HW2.relu import *
from deep.HW2.sgd import *
from deep.HW2.sigmoid import *
from deep.HW2.softmax_crossentropy import *

"""""
np.random.seed(22)
N = 5
d = 6
x = np.random.randn(N, d)
relu = ReLU('test')
print('Relu Test Cell:')
output = relu.forward(x)
correct_output = ([[0., 0., 1.08179168, 0., 0., 0., ],
                    [0.9188215, 0., 0.62649346, 0., 0.02885512, 0., ],
                    [0.58775221, 0.75231758, 0., 1.05597241, 0.74775027, 1.06467659],
                    [1.52012959, 0., 1.85998989, 0., 0., 0.337325],
                    [1.04672873, 0.62914334, 0.36305909, 0.5557497, 0., 0.02369477]])

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

dx = relu.backward(np.ones((N, d), dtype=np.float32))
correct_dx = np.array([[0., 0., 1., 0., 0., 0.],
              [1., 0., 1., 0., 1., 0.],
              [1., 1., 0., 1., 1., 1.],
              [1., 0., 1., 0., 0., 1.],
              [1., 1., 1., 1., 0., 1.]])
print('Relative error backward pass:', np.linalg.norm(dx - correct_dx))


###########################################################################
#                             sigmoid Test                                #
###########################################################################
np.random.seed(22)
N = 5
d = 6
x = np.random.randn(N, d)
sigmoid = Sigmoid('test')
print('Sigmoid Test Cell:')
output = sigmoid.forward(x)
correct_output = [[0.4770287, 0.18795539, 0.74683289, 0.44045266, 0.37962761, 0.26849495],
                  [0.71480192, 0.24905997, 0.65169394, 0.36319727, 0.50721328, 0.44256287],
                  [0.64284923, 0.67968348, 0.25759572, 0.74192012, 0.6786883, 0.74358323],
                  [0.82055756, 0.18413151, 0.86529577, 0.16817555, 0.34387488, 0.58354059],
                  [0.74014623, 0.65229519, 0.58978075, 0.63546853, 0.25189151, 0.50592342]]

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

dx = sigmoid.backward(np.ones((N, d), dtype=np.float32))
correct_dx = [[0.24947232, 0.15262816, 0.18907352, 0.24645411, 0.23551049, 0.19640541],
              [0.20386014, 0.1870291, 0.22698895, 0.23128501, 0.24994797, 0.24670098],
              [0.2295941, 0.21771385, 0.19124017, 0.19147466, 0.21807049, 0.19066721],
              [0.14724285, 0.1502271, 0.116559, 0.13989254, 0.22562495, 0.24302097],
              [0.19232979, 0.22680617, 0.24193942, 0.23164828, 0.18844218, 0.24996491]]

print('Relative error backward pass:', np.linalg.norm(dx - correct_dx))

"""""
###########################################################################
#                  Softmax with Cross Entropy Test                        #
###########################################################################
np.random.seed(22)
N = 5
d = 6

x = np.linspace(1000, 1015, num=N * d).reshape(N, d)
y = np.random.randint(0, d, (N,))

softmax_ce = SoftmaxCrossentropy('test')
print('Softmax with Cross Entropy Test Cell:')
loss, _ = softmax_ce.forward(x, y=y)
dx = softmax_ce.backward()

correct_loss = 1.6883967462546619
print('Loss relative error:', np.abs(loss - correct_loss))

correct_dx = [[0.00636809, 0.0106818, 0.01791759, 0.03005485, 0.05041383, -0.11543615],
              [0.00636809, 0.0106818, 0.01791759, 0.03005485, -0.14958617, 0.08456385],
              [0.00636809, 0.0106818, 0.01791759, 0.03005485, -0.14958617, 0.08456385],
              [-0.19363191, 0.0106818, 0.01791759, 0.03005485, 0.05041383, 0.08456385],
              [0.00636809, 0.0106818, 0.01791759, 0.03005485, -0.14958617, 0.08456385]]
print('Gradient relative error:', np.linalg.norm(dx - correct_dx))
"""""
###########################################################################
#                         Dense Test                            #
###########################################################################
np.random.seed(22)
D = 4
K = 3
N = 5
x = np.random.randn(N, D)
dense = Dense('test', D, K, l2_coef=1.)
output = dense.forward(x)

correct_output = [[-0.51242952, -1.47921276, -2.32943713],
                  [-1.17901283, -2.60908172, 0.54809823],
                  [0.74600461, -2.24752841, -1.1013558],
                  [0.75284837, 1.80111973, -2.27011589],
                  [2.03171234, -3.05396933, 1.35213333]]

print('Relative error forward pass:', np.linalg.norm(output - correct_output))

dout = np.random.randn(N, K)
dx = dense.backward(dout)

correct_dx = [[-0.25519113, -0.09724317, 0.280189, 0.87644613],
              [1.20379991, -0.78816259, -1.27930227, -4.1952743],
              [-0.77808532, -0.05005675, -3.14028536, -8.02818572],
              [0.95446653, -1.90375857, 1.62080372, 3.57597736],
              [2.86716776, -1.39892213, 0.31786772, -0.88234943]]
print('Relative error dx:', np.linalg.norm(dx - correct_dx))

correct_dW = [[3.33629487, -4.43357113, -1.89100503],
              [1.31103323, 2.17687036, -2.33906146],
              [1.69538051, -0.89256682, -0.86018824],
              [-0.87944724, 7.48073741, -7.0605863]]
print('Relative error dw:', np.linalg.norm(dense.dW - correct_dW))

correct_db = [-1.02223284, -3.61915576, -0.16696389]
print('Relative error db:', np.linalg.norm(dense.db - correct_db))

###########################################################################
#            Batch-Normalization forward pass Test (Train)                #
###########################################################################
np.random.seed(22)
N = 5
D = 4

x = np.random.randn(N, D)

batchnorm = BatchNormalization('test', D)
batchnorm.train()

x_normal = batchnorm.forward(x)

correct_x_normal = [[-0.36934148, 2.60787468, -0.04804626, 0.61796809],
                    [-1.90650387, 1.86085579, 0.06676021, 0.28590773],
                    [2.3972458, 1.14675911, 0.69370866, 0.62125601],
                    [2.24806073, -0.98185142, 1.45971198, 1.11561189],
                    [2.86418245, -1.48792145, -0.35683912, 0.13800476]]

print('Realtive error normalized x:', np.linalg.norm(x_normal - correct_x_normal))



###########################################################################
#            Batch-Normalization forward pass Test (Test)                 #
###########################################################################
np.random.seed(22)

N, D1, D2 = 5, 4, 3
W1 = np.random.randn(D1, D2)

batchnorm = BatchNormalization('test', D2)
batchnorm.train()

for t in range(50):
    X = np.random.randn(N, D1)
    a = np.maximum(0, np.matmul(X, W1))
    batchnorm.forward(a)

batchnorm.test()
X = np.random.randn(N, D1)
a = np.maximum(0, np.matmul(X, W1))
a_norm = batchnorm.forward(a)

correct_a_norm = [[2.67793539, 0.11029149, -2.26691758],
                  [0.58280269, 1.69500735, 1.48454034],
                  [0.79307162, 0.11029149, 0.94098477],
                  [0.82878218, 0.31966212, 1.95469806],
                  [0.58280269, 0.64530383, 1.08884352]]
print('Relative error:', np.linalg.norm(a_norm - correct_a_norm))


###########################################################################
#              Batch-Normalization backward pass Test                     #
###########################################################################
np.random.seed(22)
N, D = 5, 4
x = 2 * np.random.randn(N, D) + 10

batchnorm = BatchNormalization('test', D)
batchnorm.train()

dout = np.random.randn(N, D)
out = batchnorm.forward(x)
dx = batchnorm.backward(dout)

correct_dx = [[-1.31204675, -0.02199192, -0.94266767, -0.44927898],
              [0.68352166, -0.01100818, 0.23785382, 0.09507173],
              [0.34697892, 0.02983054, -0.11237967, -0.1803218],
              [1.33026886, 0.09552155, 0.16976962, 0.29533059],
              [-1.04872269, -0.09235199, 0.6474239, 0.23919846]]

print('Relative error dx:', np.linalg.norm(dx - correct_dx))


###########################################################################
#                           Drop-out Test                                 #
###########################################################################
np.random.seed(42)
N = 5
D = 5
x = np.random.randint(0, 6, (N,D)).astype(np.float32)
p = 0.5
dropout = Dropout('test', p)
dropout.train()
out = dropout.forward(x)

correct_out =  [[6., 8., 0., 8., 8.],
                [0., 4., 4., 0., 8.],
                [0., 4., 0., 0., 2.],
                [6., 0., 0., 2., 6.],
                [8., 0., 0., 0., 0.]]

dout = np.ones((N,D))
dx = dropout.backward(dout)

correct_dx =  [[2., 2., 0., 2., 2.],
               [0., 2., 2., 0., 2.],
               [0., 2., 0., 0., 2.],
               [2., 0., 0., 2., 2.],
               [2., 2., 0., 0., 0.]]

x = np.random.randint(0, 6, (N,D)).astype(np.float32)
p = 0.5
dropout = Dropout('test', p)
dropout.test()
out_test = dropout.forward(x)
correct_out_test= [[2., 5., 0., 3., 1.],
                   [3., 1., 5., 5., 5.],
                   [1., 3., 5., 4., 1.],
                   [1., 3., 1., 1., 5.],
                   [3., 5., 5., 3., 0.]]
print('Relative error Train forward pass output:', np.linalg.norm(out - correct_out))
print('Relative error Train forward pass dx:', np.linalg.norm(dx - correct_dx))
print('Relative error Test forward pass output:', np.linalg.norm(out_test - correct_out_test))


###########################################################################
#                           SGD+Momentum Test                             #
###########################################################################
N, D = 5, 4
np.random.seed(22)
dense = Dense('test', N, D, l2_coef=1.)
dense.dW = np.random.randn(N, D)
dense.db = np.random.randn(D,)

sgd = SGD(1e-2)
sgd.velocities['test'] = {'W': np.random.randn(N, D), 'b': np.zeros_like(dense.b)}

sgd.update(dense)

correct_W = [[-1.04578269, -2.09413292,  1.74896632,  0.23833633],
             [-1.13969324, -0.50236489,  1.28289011, -1.20095538],
             [-0.2181534,  -0.12424752, -1.3418189,   0.13508095],
             [ 0.59827594, -0.35371713, -2.00549095,  3.3314237 ],
             [-1.09332467,  1.15610425,  1.24786695, -1.06838115],]
correct_b = [ 1.86566377, -1.59381353, -0.62684131,  0.33332912]

print('W Relative error:', np.linalg.norm(correct_W - dense.W))
print('b Relative error: ', np.linalg.norm(correct_b - dense.b))
"""""


###########################################################################
#                                ADAM Test                                #
###########################################################################
N, D = 5, 4
dense = Dense('test', N, D, l2_coef=1.)
dense.W = np.linspace(-1, 1, N * D).reshape(N, D)
dense.dW = np.linspace(-1, 1, N * D).reshape(N, D)
dense.db = np.zeros(D, )
adam = Adam(1e-2)

m = np.linspace(0.6, 0.9, N * D).reshape(N, D)
v = np.linspace(0.7, 0.5, N * D).reshape(N, D)

adam.m['test'] = {'W': m, 'b': np.zeros(D, )}
adam.v['test'] = {'W': v, 'b': np.zeros(D, )}
adam.iteration_number = 6
adam.update(dense)

next_param = dense.W

correct_next_param = [[-1.00086812, -0.89566086, -0.79045452, -0.68524913],
                      [-0.58004471, -0.47484131, -0.36963895, -0.26443768],
                      [-0.15923753, -0.05403855, 0.05115923, 0.15635575],
                      [0.26155096, 0.36674482, 0.47193728, 0.57712826],
                      [0.68231771, 0.78750557, 0.89269177, 0.99787623]]
correct_v = [[0.7003, 0.68958476, 0.67889169, 0.66822078],
             [0.65757202, 0.64694543, 0.636341, 0.62575873],
             [0.61519861, 0.60466066, 0.59414488, 0.58365125],
             [0.57317978, 0.56273047, 0.55230332, 0.54189834],
             [0.53151551, 0.52115485, 0.51081634, 0.5005]]
correct_m = [[0.44, 0.46473684, 0.48947368, 0.51421053],
             [0.53894737, 0.56368421, 0.58842105, 0.61315789],
             [0.63789474, 0.66263158, 0.68736842, 0.71210526],
             [0.73684211, 0.76157895, 0.78631579, 0.81105263],
             [0.83578947, 0.86052632, 0.88526316, 0.91]]

print('W error: ', np.linalg.norm(correct_next_param - next_param))
print('v error: ', np.linalg.norm(correct_v - adam.v['test']['W']))
print('m error: ', np.linalg.norm(correct_m - adam.m['test']['W']))
