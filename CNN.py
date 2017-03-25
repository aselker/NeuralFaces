#Import Everything
import scipy.io
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import timeit


#Set Parameters
lrnrate = .01
reg = 1e-5
rng = np.random.RandomState(55764)

#Pull in data from Databases (Larger General Face Data and Class Data)
mat = scipy.io.loadmat('face_detect.mat')
face_data = mat['faces_train']
print(face_data.shape)


# Change Faces to 4D Tensor
face_data = np.expand_dims(face_data, axis=3)

face_data = face_data.swapaxes(0,2).swapaxes(1,3)
print(face_data.shape)

# instantiate 4D tensor for input
input_data = T.tensor4(name='input_data')

# initialize shared variable for weights.
w_shp = (2, 1, 9, 9)
w_bound = np.sqrt(1* 9 * 9)
W = theano.shared(np.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input_data.dtype), name ='W')

b_shp = (2,)
b = theano.shared(np.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input_data.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input_data, W)

# Network the Convolution
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input_data], output)

print(f(face_data[:,:,:,:]).shape)

#Possibly Pass Faces through Eigen Face Process to get less intricate data and speed up computation


#Possibly go through autoencoding process?



######## MACHINE LEARNING SETUP ########

##### Create Conv Net Class
# Setup Feed Forward Network


# Setup Cost Function


# Setup Gradient Function


##### Create Normal Net Class
# Setup Feed Forward Network


# Setup Cost Function


# Setup Gradient Function


######## MACHINE LEARNING RUNNING ########

### Split Data into Training and Testing set / Use k-means Cross Validation in order to allow for parameterization and choosing good values for training


# START TIMEIT
# Train Network


# END TIMEIT
# Find Training Accuracy


# Use Accuracy of Testing Data in order to decide how to parameterize the Machine LEARNING

# START TIMEIT
# Test Data on Actual Testing Data to get an idea of how well it does


# END TIMEIT
