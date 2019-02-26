# Adapted form the on the MNIST expert tutorial by Google, this module aims to train 
# the Multilayer Perceptron by using the MNIST database of handwritten digits.
#
# This program was based on the following programs:
#
# [1] -   https://www.tensorflow.org/get_started/mnist/beginners
# [2] -   https://www.tensorflow.org/get_started/mnist/pros
# [3] -   https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py
# [4] -   https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================================================================================

# Note: The used Tensorflow was not compiled to use SSE 4.2 / AVX / AVX2 / FMA

# Import modules
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# Parameter
Model_FULL_PATH = os.path.join(os.path.dirname(__file__), 'Model', 'Model_NN_Intermediate.ckpt')
FModel_FULL_PATH = os.path.join(os.path.dirname(__file__), 'Model', 'Model_NN.ckpt')

# Disable CPU-related notifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration for the GPU to avoid the crash
Config = tf.ConfigProto()
Config.gpu_options.allow_growth = True

# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
Acc = tf.Variable(1.0)
AccO = tf.Variable(1.0)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv1_2 = weight_variable([3, 3, 32, 32])
b_conv1_2 = bias_variable([32])
h_conv1_2 = tf.nn.relu(conv2d(h_conv1, W_conv1_2) + b_conv1_2)

h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv2_2 = weight_variable([3, 3, 64, 64])
b_conv2_2 = bias_variable([64])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2, W_conv2_2) + b_conv2_2)

h_pool2 = max_pool_2x2(h_conv2_2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# Initialize the variables
init_op = tf.global_variables_initializer()

# Running the training session
sess.run(init_op)

# Train the model and save the model to disk as a file of the .ckpt extension
# File is stored in the same directory as this python script is started

print ("\n***** Training a prediction model... *****\n")
print ("Model: Convolutional Neural Network, Activation: ReLU + Softmax\n")
for i in range(20000):
  average_cost = 0
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("Step = %05d, Training accuracy = %.2f" % (i, train_accuracy))

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Update the accuracy to the variable 'Acc'
AccOp = Acc.assign(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.run(AccOp)

save_path = saver.save(sess, Model_FULL_PATH)
print ("\n***** Training complete! *****\n")

# Run another session for the optmization purpose
print ("***** Running an optimization... *****\n")
sess.run(init_op)

# Restore the previous model
saver.restore(sess, Model_FULL_PATH)

# Resume the training
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("Step = %05d, Training accuracy = %.2f, "%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#	Update the accuracy to the variable 'AccO'
AccOp = AccO.assign(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.run(AccOp)
  
save_path_opt = saver.save(sess, FModel_FULL_PATH)
print ("\n***** Optimization complete! *****\n")
print ("Model saved in file: ", save_path_opt)
print ("***** Accuracy of the Model (Before Optimization) = ", sess.run(Acc), "(After Optimation) = ", sess.run(AccO))