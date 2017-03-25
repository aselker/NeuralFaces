import scipy.io
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import timeit


input_data = T.dtensor4('input_data')
maxpool_shape = (2, 2)
pool_out = pool.pool_2d(input_data, maxpool_shape, ignore_border=True)
f = theano.function([input_data],pool_out)

invals = np.random.RandomState(1).rand(3, 2, 5, 5)
print('With ignore_border set to True:')
print('invals[0, 0, :, :] =\n', invals[0, 0, :, :])
print('output[0, 0, :, :] =\n', f(invals)[0, 0, :, :])

pool_out = pool.pool_2d(input_data, maxpool_shape, ignore_border=False)
f = theano.function([input_data],pool_out)
print('With ignore_border set to False:')
print('invals[1, 0, :, :] =\n ', invals[1, 0, :, :])
print('output[1, 0, :, :] =\n ', f(invals)[1, 0, :, :])
