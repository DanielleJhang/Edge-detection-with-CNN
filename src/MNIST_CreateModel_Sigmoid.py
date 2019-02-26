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

# Disable CPU-related notifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.sigmoid(tf.matmul(x, W) + b)
Acc = tf.Variable(1.0)

# Functions
def Get_Train_Data(num):
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    return x_train, y_train

def Get_Test_Data(num):
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test

# Parameters
x_train, y_train = Get_Train_Data(5500)
x_test, y_test = Get_Test_Data(10000)
Learning_Rate = 0.05

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# Train the model and save the model to disk as a file of the .ckpt extension
# File is stored in the same directory as this python script is started

print ("\n***** Training a prediction model... *****\n")
print ("Model: Single-layer Perceptron, Activation: Sigmoid\n")
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(4000 + 1):
        sess.run(train_step, feed_dict={x: x_train, y_: y_train})

        if i % 100 == 0 :
            print("Step = %04d" % i, ", Training accuracy = ", sess.run(accuracy, {x: x_test, y_: y_test}), ", Average Cost = ", sess.run(cross_entropy, {x: x_train, y_: y_train}))

	#	Update the accuracy to the variable 'Acc'
    AccOp = Acc.assign(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.run(AccOp)

    save_path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'Model', 'Model_Simple.ckpt'))
    
    print ("\n***** Training complete! *****\n")
    print ("Model saved in file: ", save_path)
    print ("Model's accuracy : ", Acc.eval())