import scipy.io
import numpy
import theano
import scipy.misc
from matplotlib import pyplot as plt
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from NetworkClasses import ConvPoolNet, HiddenLayer, LogisticRegression
import timeit
#import random

learning_rate = 1
reg = 1e-5
trainsteps = 3
rng = numpy.random.RandomState(55764)
nkerns = (2,4)
face_count = 1

#Pull in data from Databases (Larger General Face Data and Class Data)
mat = scipy.io.loadmat('face_detect.mat')
#print(mat)
face_data = mat['faces_train']
print('Face Data Shape' + str(face_data.shape))
names_train = mat['names_train']
train_names = names_train

rotated_train_names = [['\x00' for i in range(len(train_names))] for j in range(max([len(train_names[i]) for i in range(len(train_names)-1)]))]

for i in range(len(train_names)):
    for j in range(len(train_names[i])):
        rotated_train_names[j][i] = train_names[i][j]

new_train_names = [""] * len(rotated_train_names)

for i in range(len(rotated_train_names)):
    for j in range(len(rotated_train_names[i])):
        if rotated_train_names[i][j] == '\x00':
            break
        else:
            new_train_names[i] += rotated_train_names[i][j]
cur = 0
num = -1
for i in range(len(new_train_names)): #Use names and convert them all to integer numbers denoting what class they are (each class is a person)
    if not new_train_names[i] == cur:
        cur = new_train_names[i]
        num += 1
    new_train_names[i] = num
train_names = new_train_names
print('train_names shape: ' + str(len(train_names)))
#print(train_names)

# Change Faces to 4D Tensor
face_data = numpy.expand_dims(face_data, axis=3)

face_data = face_data.swapaxes(0,2).swapaxes(1,3)

############################################# GET TEST DATA
faces_test_easy = mat['faces_test_easy']
names_test_easy = mat['names_test_easy']
faces_test_hard = mat['faces_test_hard']
names_test_hard = mat['names_test_hard']
test_names_easy = names_test_easy

rotated_test_names_easy = [['\x00' for i in range(len(test_names_easy))] for j in range(max([len(test_names_easy[i]) for i in range(len(test_names_easy)-1)]))]

for i in range(len(test_names_easy)):
    for j in range(len(test_names_easy[i])):
        rotated_test_names_easy[j][i] = test_names_easy[i][j]

new_test_names_easy = [""] * len(rotated_test_names_easy)

for i in range(len(rotated_test_names_easy)):
    for j in range(len(rotated_test_names_easy[i])):
        if rotated_test_names_easy[i][j] == '\x00':
            break
        else:
            new_test_names_easy[i] += rotated_test_names_easy[i][j]
cur = 0
num = -1
for i in range(len(new_test_names_easy)): #Use names and convert them all to integer numbers denoting what class they are (each class is a person)
    if not new_test_names_easy[i] == cur:
        cur = new_test_names_easy[i]
        num += 1
    new_test_names_easy[i] = num
test_names_easy = new_test_names_easy
print('test_names_easy shape: ' + str(len(test_names_easy)))
#print(train_names)


# Change Faces to 4D Tensor
faces_test_easy = numpy.expand_dims(faces_test_easy, axis=3)

faces_test_easy = faces_test_easy.swapaxes(0,2).swapaxes(1,3)

#########################################
test_names_hard = names_test_hard

rotated_test_names_hard = [['\x00' for i in range(len(test_names_hard))] for j in range(max([len(test_names_hard[i]) for i in range(len(test_names_hard)-1)]))]

for i in range(len(test_names_hard)):
    for j in range(len(test_names_hard[i])):
        rotated_test_names_hard[j][i] = test_names_hard[i][j]

new_test_names_hard = [""] * len(rotated_test_names_hard)

for i in range(len(rotated_test_names_hard)):
    for j in range(len(rotated_test_names_hard[i])):
        if rotated_test_names_hard[i][j] == '\x00':
            break
        else:
            new_test_names_hard[i] += rotated_test_names_hard[i][j]
cur = 0
num = -1
for i in range(len(new_test_names_hard)): #Use names and convert them all to integer numbers denoting what class they are (each class is a person)
    if not new_test_names_hard[i] == cur:
        cur = new_test_names_hard[i]
        num += 1
    new_test_names_hard[i] = num
test_names_hard = new_test_names_hard
print('test_names_hard shape: ' + str(len(test_names_hard)))
#print(train_names)


# Change Faces to 4D Tensor
faces_test_hard = numpy.expand_dims(faces_test_easy, axis=3)

faces_test_hard = faces_test_easy.swapaxes(0,2).swapaxes(1,3)

#########################################

#batch_size = face_data.shape[0]//2

#index = T.iscalar('index')#theano.shared(value = 0, name = 'index')
x = T.matrix('x')   # the data is presented as rasterized images
y = T.lvector('y')  # the labels are presented as 1D vector of
                    # [int] labels
#face_count = x.shape[0]

#datasets = load_data(dataset)

### MAKE IT WORK WITH A TESTING EVENTUALLY ###
train_set_x, train_set_y = face_data, train_names
test_easy_set_x, test_easy_set_y = faces_test_easy, test_names_easy
test_hard_set_x, test_hard_set_y = faces_test_hard, test_names_hard
print('--------------------' + str(train_set_x.shape))

# compute number of minibatches for training, validation and testing
#n_train_batches = train_set_x.shape[0] // batch_size
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
#n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

######################
# BUILD ACTUAL MODEL #
######################
print('Building Model')

# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images.
im_shape = (face_count, 1, 256, 256)
layer0_input = x.reshape(im_shape) # get input shape

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

layer0 = ConvPoolNet(
    rng,
    input=layer0_input,
    image_shape=im_shape,
    filter_shape=(nkerns[0], 1, 10, 10),
    poolsize=(2, 2)
)

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
layer1 = ConvPoolNet(
    rng,
    input=layer0.output,
    image_shape=(face_count, nkerns[0], (256-10)//2, (256-10)//2),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    poolsize=(2, 2)
)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=int(nkerns[1] * ((256-10)//2-5)//2 * ((256-10)//2-5)//2),
    n_out=face_count,
    activation=T.tanh
)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=face_count, n_out=40)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y) ## Add Regularization

print("Building Functions")

# create a function to compute the mistakes that are made by the model
test_model = theano.function(
    [x,y],
    layer3.errors(y),
)

"""validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)"""

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params
#print('layer0:')
#print(type(layer0.params))
#print(layer0.params[0].get_value().shape)

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

#print(type(int(index)))

train_model = theano.function(
    [x,y],
    [layer0.W[0][0],cost],
    updates=updates
    #givens={
    #    x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #    y: train_set_y[index * batch_size: (index + 1) * batch_size]
    #}
)

print(train_set_x.shape)
print(test_easy_set_x.shape)
print("Training")
kernel = 0
for i in range(trainsteps):
    for j in range(len(train_set_x)//face_count):
        #index.set_value(index.get_value()+1)
        #print(train_set_x[j*face_count:(j+1)*face_count].shape)
        #print(train_set_x[j][0].shape)
        kernel, cost = train_model(train_set_x[j][0], numpy.expand_dims(train_set_y[j],axis=0))#train_model(train_set_x[j*face_count:(j+1)*face_count], train_set_y[j*face_count:(j+1)*face_count])
    print('Cost = ' + str(cost))
    scipy.misc.imsave('kernel'+str(i)+'.png', kernel)

print("\nPredicting\n")

predict = []
for j in range(len(test_easy_set_x)):
    #index.set_value(index.get_value()+1)
    print(j)
    predict.append(test_model(test_easy_set_x[j][0], numpy.expand_dims(test_easy_set_y[j],axis=0)))
print("%i/%i"%(len(predict)-sum(predict),len(predict)))

print("Done")
