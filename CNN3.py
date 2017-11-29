
# coding: utf-8

# # Tensorflow Demo: MNIST for Experts
# 
# Before start using this, please select `Cell` - `All Output` - `Clear` to clear the old results. See [TensorFlow Tutorial](https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html) for details of the tutorial.

# # Loading MNIST training data
# 

# In[1]:

# Import tensorflow
import tensorflow as tf
import os
import numpy as np

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import base
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import matplotlib.pyplot as plt
from skimage.io import imread


# In[ ]:




# In[2]:

# class DataSet(object):

#     def __init__(self,
#                images,
#                labels,
#                fake_data=False,
#                one_hot=False,
#                dtype=dtypes.float32,
#                reshape=True,
#                seed=None):
#         """Construct a DataSet.
#         one_hot arg is used only if fake_data is true.  `dtype` can be either
#         `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
#         `[0, 1]`.  Seed arg provides for convenient deterministic testing."""
#         seed1, seed2 = random_seed.get_seed(seed)
#         # If op level seed is not set, use whatever graph level seed is returned
#         np.random.seed(seed1 if seed is None else seed2)
#         dtype = dtypes.as_dtype(dtype).base_dtype
#         if dtype not in (dtypes.uint8, dtypes.float32):
#             raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
#                           dtype)
#         if fake_data:
#             self._num_examples = 10000
#             self.one_hot = one_hot
#         else:
#             assert images.shape[0] == labels.shape[0], (
#             'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
#             self._num_examples = images.shape[0]

#             # Convert shape from [num examples, rows, columns, depth]
#             # to [num examples, rows*columns] (assuming depth == 1)
#             if reshape:
#                 assert images.shape[3] == 1
#                 images = images.reshape(images.shape[0],
#                             images.shape[1] * images.shape[2])
#             if dtype == dtypes.float32:
#                 # Convert from [0, 255] -> [0.0, 1.0].
#                 #images = images.astype(np.float32)
#                 images = np.multiply(images, 1.0 / 255.0)
#         self._images = images
#         self._labels = labels
#         self._epochs_completed = 0
#         self._index_in_epoch = 0

#     @property
#     def images(self):
#         return self._images

#     @property
#     def labels(self):
#         return self._labels

#     @property
#     def num_examples(self):
#         return self._num_examples

#     @property
#     def epochs_completed(self):
#         return self._epochs_completed

#     def next_batch(self, batch_size, fake_data=False, shuffle=True):
#         """Return the next `batch_size` examples from this data set."""
#         if fake_data:
#             fake_image = [1] * 6020
#             if self.one_hot:
#                 fake_label = [1] + [0] * 9
#             else:
#                 fake_label = 0
#             return [fake_image for _ in xrange(batch_size)], [
#                 fake_label for _ in xrange(batch_size)]
#         start = self._index_in_epoch
#         # Shuffle for the first epoch
#         if self._epochs_completed == 0 and start == 0 and shuffle:
#             perm0 = np.arange(self._num_examples)
#             np.random.shuffle(perm0)
#             self._images = self.images[perm0]
#             self._labels = self.labels[perm0]
#         # Go to the next epoch
#         if start + batch_size > self._num_examples:
#             # Finished epoch
#             self._epochs_completed += 1
#             # Get the rest examples in this epoch
#             rest_num_examples = self._num_examples - start
#             images_rest_part = self._images[start:self._num_examples]
#             labels_rest_part = self._labels[start:self._num_examples]
#             # Shuffle the data
#             if shuffle:
#                 perm = np.arange(self._num_examples)
#                 np.random.shuffle(perm)
#                 self._images = self.images[perm]
#                 self._labels = self.labels[perm]
#             # Start next epoch
#             start = 0
#             self._index_in_epoch = batch_size - rest_num_examples
#             end = self._index_in_epoch
#             images_new_part = self._images[start:end]
#             labels_new_part = self._labels[start:end]
#             return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
#         else:
#             self._index_in_epoch += batch_size
#             end = self._index_in_epoch
#             return self._images[start:end], self._labels[start:end]


# def read_data_sets(fake_data=False,
#                    one_hot=False,
#                    dtype=dtypes.float32,
#                    reshape=True,
#                    validation_size=5000,
#                    seed=None):
#     if fake_data:

#         def fake():
#             return DataSet(
#               [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

#         train = fake()
#         validation = fake()
#         test = fake()
#         return base.Datasets(train=train, validation=validation, test=test)
    
#     TRAIN = '/data/face_dataset/train/'
#     TEST = '/data/face_dataset/test/'
#     def read_images(direc, test=False):
        
#         pids = os.listdir(direc)
#         images = [] 
#         labels = []
#         for pid in pids:
#             if test: 
#                 labels.append(pid) #TODO PID is a filename here 
#                 images.append(imread(direc + pid, dtype=np.uint8))
#             else: 
#                 for image_file in os.listdir(direc + pid):
#                     #imagefiles.append(image_file)
#                     labels.append(pid)
#                     images.append(imread(direc + pid + '/' + image_file))
                    
#         #print(images)
#         return np.array(images), np.array(labels)
    

#     train_images, train_labels = read_images(TRAIN)
#     test_images, test_labels = read_images(TEST, test=True)
                              
    
                
#     if not 0 <= validation_size <= len(train_images):
#         raise ValueError(
#             'Validation size should be between 0 and {}. Received: {}.'
#             .format(len(train_images), validation_size))

#     validation_images = train_images[:validation_size]
#     validation_labels = train_labels[:validation_size]
#     train_images = train_images[validation_size:]
#     train_labels = train_labels[validation_size:]


#     options = dict(dtype=dtypes.uint8, reshape=False, seed=seed)

#     train = DataSet(train_images, train_labels, **options)
#     validation = DataSet(validation_images, validation_labels, **options)
#     test = DataSet(test_images, test_labels, **options)

#     return base.Datasets(train=train, validation=validation, test=test)


# In[3]:

# data = read_data_sets()


# In[4]:

# mnist = data


# In[5]:

# data.train.images


# In[6]:

TRAIN = '/data/face_dataset/train/'
TEST = '/data/face_dataset/test/'
def read_images(direc, test=False):

    pids = os.listdir(direc)
    images = [] 
    labels = []
    for pid in pids:
        if test: 
            labels.append(pid) #TODO PID is a filename here 
            images.append(direc + pid)
        else: 
            for image_file in os.listdir(direc + pid):
                #imagefiles.append(image_file)
                labels.append(int(pid))
                images.append(direc + pid + '/' + image_file)

    #print(images)
    return np.array(images), np.array(labels)


train_images, train_labels = read_images(TRAIN)
test_images, test_labels = read_images(TEST, test=True)


# In[7]:

#Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
label_order = train_labels.copy()
label_order = set(label_order)
label_order = sorted(list(label_order))

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decodes = tf.divide(tf.to_float(image_decoded), tf.constant(255.0))
    #print(image_decoded)
    #image_resized = tf.image.resize_images(image_decoded, [64, 64])
#     y_arr = np.zeros([398])
#     ind = label_order.index(label)
#     y_arr[ind] = 1
#     print(y_arr)
    
    return tf.reshape(image_decoded, [-1, 12288]), label

# A vector of filenames.
#filenames = tf.constant(imagefiles)

# `labels[i]` is the label for the image in `filenames[i].
#labels = tf.constant(labels)

dataset_train = tf.data.Dataset.from_tensor_slices((tf.constant(train_images), tf.constant(train_labels)))
dataset_train = dataset_train.map(_parse_function)

dataset_test = tf.data.Dataset.from_tensor_slices((tf.constant(test_images), tf.constant(test_labels)))
dataset_test = dataset_test.map(_parse_function) 


# In[8]:

#mnist = dataset_train



# # Build a Multilayer Convolutional Network
# 
# Getting 91% accuracy on MNIST is bad. It's almost embarrassingly bad. In this section, we'll fix that, jumping from a very simple model to something moderately sophisticated: a small convolutional neural network. This will get us to around 99.2% accuracy -- not state of the art, but respectable.

# In[9]:

x = tf.placeholder(tf.float32, [None, 12288])
W = tf.Variable(tf.zeros([12288, 398]))
b = tf.Variable(tf.zeros([398]))
y_ = tf.placeholder(tf.float32, [None, 398])


# ## Weight & Biases Initialization
# 
# To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons." Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.

# In[10]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ## Convolution & Pooling
# 
# TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
# 
# ![](http://deeplearning.stanford.edu/wiki/images/6/6c/Convolution_schematic.gif)
# 
# http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution
# 
# Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations into functions.
# 
# ![](http://www.wildml.com/wp-content/uploads/2015/11/Screen-Shot-2015-11-05-at-2.18.38-PM.png)
# 
# ![](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/img/Conv-9-Conv2Max2Conv2.png)
# 
# http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

# In[11]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# ## First Convolutional Layer
# 
# We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.

# In[12]:

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])


# To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

# In[13]:

x_image = tf.reshape(x, [-1,64,64,3])


# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

# In[14]:

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# ## Second Convolutional Layer
# 
# In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

# In[15]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# ## Densely Connected Layer
# 
# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

# In[16]:

W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# ### Dropout
# 
# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

# In[17]:

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #don't start with this, its hard 


# ## Readout Layer
# 
# Finally, we add a softmax layer, just like for the one layer softmax regression.

# In[18]:

W_fc2 = weight_variable([1024, 398])
b_fc2 = bias_variable([398])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# ## Train and Evaluate the Model
# 
# How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above. The differences are that: we will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer; we will include the additional parameter keep_prob in feed_dict to control the dropout rate; and we will add logging to every 100th iteration in the training process.
# 
# 

# In[ ]:

# initialize variables and session
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())
    
BATCH_SIZE = 100 

batch = dataset_train.batch(BATCH_SIZE) #TODO 
itera = batch.make_one_shot_iterator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Run mini-batch training on 100 elements 20000 times.
    for i in range(20000):
        t = sess.run(itera.get_next())
        #print(t[0][0], t[1])

        #encoding 
        y_arr = np.zeros([BATCH_SIZE, 398])
        for j in range(BATCH_SIZE):
            ind = label_order.index(t[1][j])
            y_arr[j][ind] = 1
            #print(j)
            #y_arr = np.reshape(y_arr, [-1, 398])
        #print(t[0].reshape(BATCH_SIZE, 12288)/255.0)
        if i%10 == 0:
            #print(batch[0])
            train_accuracy = accuracy.eval(feed_dict={
                  x:t[0].reshape(BATCH_SIZE, 12288)/255.0, y_: y_arr, keep_prob: 1.0})
            #print(i)
#             train_accuracy = sess.run(accuracy, feed_dict={
#                  x:t[0].reshape(BATCH_SIZE, 12288), y_: y_arr, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            #print(i)
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #sess.run(itera.get_next())
#         sess.run(train_step, feed_dict={x: t[0].reshape(BATCH_SIZE, 12288) , y_: y_arr, keep_prob: 0.5})
        train_step.run(feed_dict={x: t[0].reshape(BATCH_SIZE, 12288)/255.0 , y_: y_arr, keep_prob: 0.5})
        #print(i, "Completed epoch")
    # todo run the test data 
    # plug it in 



    #print("test accuracy %g" % sess.run(accuracy, feed_dict={
    #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[ ]:




# The final test set accuracy after running this code should be approximately 99.2%.
# 
# We have learned how to quickly and easily build, train, and evaluate a fairly sophisticated deep learning model using TensorFlow.
# 
# 1: For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks.

# # Visualizing with TensorBoard
# 
# Visualize with [TensorBoard](https://www.tensorflow.org/tensorboard/index.html).

# ![](https://www.tensorflow.org/versions/master/images/mnist_tensorboard.png)

# In[ ]:




# In[ ]:



