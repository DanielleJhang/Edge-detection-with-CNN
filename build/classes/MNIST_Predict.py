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
import sys
import os
import tensorflow as tf
from PIL import Image,ImageFilter

# Disable CPU-related notifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def Predict_Digit(imvalue):
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    Acc = tf.Variable(1.0)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    print("Loading the model...")
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(os.path.dirname(__file__), 'Model', 'Model_Simple.ckpt'))
        print ("Model successfully loaded. Current Model = Single-layer Perceptron, Activation: Softmax")
        print ("Current Model's accuracy = ", sess.run(Acc))
   
        prediction=tf.argmax(y,1)
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

def Process_Image(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
	
	# Creates white canvas of 28x28 pixels
    newImage = Image.new('L', (28, 28), (255))
    
	# Conditinal to check whether the input image's dimensions are bigger than the standard
    if width > height: 
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height),0)) #resize height according to ratio width
        
		# Rare case but minimum is 1 pixel
        if (nheight == 0): 
            nheight = 1  

        # Resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		
		# Calculate horizontal pozition
        wtop = int(round(((28 - nheight) / 2),0))
		
		# Paste resized image on white canvas
        newImage.paste(img, (4, wtop)) 
    else:
        # Height is bigger. Heigth becomes 20 pixels
		# Resize width according to ratio height		
        nwidth = int(round((20.0 / height * width),0))
        
        if (nwidth == 0):
            nwidth = 1
        
		# Resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2),0)) #caculate vertical pozition
		
		# Paste resized image on white canvas
        newImage.paste(img, (wleft, 4))

	# Get pixel values
    tv = list(newImage.getdata()) 
    
    # Normalize pixels to 0 and 1. 0 is pure white, 1 is pure black
    tva = [(255 - x) * 1.0 / 255.0 for x in tv] 
    return tva

def main(argv):
    imvalue = Process_Image(argv)
    predint = Predict_Digit(imvalue)
    print ("Result = " + str(predint[0]))
    
if __name__ == "__main__":
    main(sys.argv[1])
