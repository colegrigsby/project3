
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
import time
from PIL import Image


# In[ ]:




# In[ ]:




# In[2]:

# data = read_data_sets()


# In[3]:

# mnist = data


# In[4]:

# data.train.images


# In[5]:

TRAIN = '/data/face_dataset/train/'
TEST = '/data/face_dataset/test/'
def read_images(direc, test=False):

    pids = os.listdir(direc)
    images = [] 
    labels = []
    for pid in pids:
        if test: 
            labels.append(pid.split('-')[0]) #TODO PID is a filename here 
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


# In[6]:

label_order = train_labels.copy()
label_order = set(label_order)
label_order = sorted(list(label_order))


# In[7]:

# #Reads an image from a file, decodes it into a dense tensor, and resizes it
# # to a fixed shape.


# def _parse_function(filename, label):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_image(image_string, dtypes=tf.float32)
#     #image_decoded = tf.divide(tf.to_float(image_decoded), tf.constant(255.0))
#     #print(image_decoded)
#     #image_resized = tf.image.resize_images(image_decoded, [64, 64])
# #     y_arr = np.zeros([398])
# #     ind = label_order.index(label)
# #     y_arr[ind] = 1
# #     print(y_arr)
    
#     return tf.reshape(image_decoded, [12288]), label

# # A vector of filenames.
# #filenames = tf.constant(imagefiles)

# # `labels[i]` is the label for the image in `filenames[i].
# #labels = tf.constant(labels)

# dataset_train = tf.data.Dataset.from_tensor_slices((tf.constant(train_images), tf.constant(train_labels)))
# dataset_train = dataset_train.map(_parse_function)

# dataset_test = tf.data.Dataset.from_tensor_slices((tf.constant(test_images), tf.constant(test_labels)))
# dataset_test = dataset_test.map(_parse_function) 


# In[8]:

def writeTfrecords(toname, filenames, labels, train=True):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    tfrecords_filename = toname

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path, label in zip(filenames, labels):

        img = np.array(Image.open(img_path))
        
        y_arr = np.zeros([398])
        if train:
            ind = label_order.index(label)
            #print(ind)
            y_arr[ind] = 1
        
        #print(y_arr.reshape(-1))

        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw),
            'labels_raw': _float32_feature(y_arr.reshape(-1))}))

        writer.write(example.SerializeToString())

    writer.close()



# In[9]:

#writeTfrecords("traindataset.tfrecords", train_images, train_labels)


# TODO shuffle 


# In[10]:

#writeTfrecords("testdataset.tfrecords", test_images, test_labels, train=False)


# In[11]:

def parse_tfrecords(example_proto):
    
    features = tf.parse_single_example(
      example_proto,
      # Defaults are not specified since both keys are required.
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'labels_raw': tf.FixedLenFeature([398], tf.float32)
        })
    
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [12288])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        
    return image, features["labels_raw"]


# In[ ]:




# # Build a Multilayer Convolutional Network
# 
# Getting 91% accuracy on MNIST is bad. It's almost embarrassingly bad. In this section, we'll fix that, jumping from a very simple model to something moderately sophisticated: a small convolutional neural network. This will get us to around 99.2% accuracy -- not state of the art, but respectable.

# In[12]:

x = tf.placeholder(tf.float32, [None, 12288])
W = tf.Variable(tf.zeros([12288, 398]))
b = tf.Variable(tf.zeros([398]))
y_ = tf.placeholder(tf.float32, [None, 398])


# ## Weight & Biases Initialization
# 
# To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons." Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.

# In[13]:

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

# In[14]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# ## First Convolutional Layer
# 
# We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.

# In[15]:

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])


# To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

# In[16]:

x_image = tf.reshape(x, [-1,64,64,3])


# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

# In[17]:

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# ## Second Convolutional Layer
# 
# In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

# In[18]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# ## Densely Connected Layer
# 
# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

# In[19]:

W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# ### Dropout
# 
# To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

# In[20]:

#TODO 
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #don't start with this, its hard 


# ## Readout Layer
# 
# Finally, we add a softmax layer, just like for the one layer softmax regression.

# In[21]:

W_fc2 = weight_variable([1024, 398])
b_fc2 = bias_variable([398])

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


# ## Train and Evaluate the Model
# 
# How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above. The differences are that: we will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer; we will include the additional parameter keep_prob in feed_dict to control the dropout rate; and we will add logging to every 100th iteration in the training process.
# 
# 

# In[22]:

def get_pred(values):
    return label_order[np.argmax(values)]


# In[23]:

BATCH_SIZE = 128

traindata = tf.data.TFRecordDataset(['traindataset.tfrecords'])
traindata = traindata.map(parse_tfrecords)
traindata = traindata.batch(BATCH_SIZE)

testdata = tf.data.TFRecordDataset(['testdataset.tfrecords'])
testdata = testdata.map(parse_tfrecords)
testdata = testdata.batch(1024)


# In[50]:

# initialize variables and session
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())
    

#batch = dataset_train.batch(BATCH_SIZE) 
#itera = batch.make_one_shot_iterator()
#dataset = dataset_train
#dataset = dataset.shuffle(buffer_size=10000) #TODO this definitely slows it down 

#dataset = dataset.repeat() #could just repeat number of epoch times and create onshot iterator 
itera = traindata.make_initializable_iterator()
next_element = itera.get_next()

itertest = testdata.make_initializable_iterator()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(itera.initializer)
        its = 0 
        start_time = time.time()
        
        print("starting epoch " + str(i))
        while True:
            try:
                t = sess.run(next_element)
                train_step.run(feed_dict={x: t[0],  y_: t[1]})
                if its % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x:t[0], y_: t[1]})
                    print("step %d, training accuracy %f" % (its, train_accuracy))
                    
                its += 1
            except tf.errors.OutOfRangeError:
                break
                
        print("testing and saving")

        predictions = []
        sess.run(itertest.initializer)
        while True:
            try: 
                tt = sess.run(itertest.get_next())
                p = sess.run(y_conv, feed_dict={x: tt[0]})
                #print(p, p.shape)
                if len(predictions):
                    predictions = np.concatenate((predictions, p))
                else:
                    predictions = p
                
            except tf.errors.OutOfRangeError:
                break

        #print(predictions)
        preds = [get_pred(p) for p in predictions] 

        #write predictions at each epoch 
        np.save('workfile' + str(i), preds)
        elapsed_time = time.time() - start_time
        print("end epoch: " + str(i) + " minutes run: " + str(elapsed_time/60))
    
    


# In[25]:

# len(label_order)


# In[44]:

# p = [[2,3], [5,4]] 
# b = [[]]
# np.concatenate((p, b))


# In[ ]:

## start_time = time.time()
# your code

elapsed_time = time.time() - start_time


# In[47]:

# np.load("workfile0.npy")


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

np.load("workfile1.npy")


# In[ ]:



