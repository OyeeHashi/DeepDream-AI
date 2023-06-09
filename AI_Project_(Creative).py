#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
#import pandas as pd
import cv2
from PIL import Image 
import matplotlib.pyplot as plt
#import seaborn as sns
#import random 
import os
import PIL.Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '4'
# In[3]:

tf.__version__


# In[4]:


# We loaded this trained model as didn't had the computational power and time to train for dataset
# 1.2 Million images and 1000 categories
from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(include_top = False, weights = 'imagenet') # cutting off last layer to extract only the features

base_model.summary()

# In[5]:


# Open the first image
img_1 = Image.open(r"C:\Users\Hashii\Desktop\mars.jpg")

# Open the second image
img_2 = Image.open(r"C:\Users\Hashii\Desktop\eiffel.jpg")

# Resize the images to the same size
img_1 = img_1.resize((512, 512))
img_2 = img_2.resize((512, 512))

# Blending the two images

image = Image.blend(img_1, img_2, 0.5) #

# Save the blended image
image.save("img_0.jpg")


# In[6]:


# Load the image
Sample_Image = tf.keras.preprocessing.image.load_img('img_0.jpg')

# Convert the image to a NumPy array
sample_image_array = tf.keras.preprocessing.image.img_to_array(Sample_Image)


# In[7]:


Sample_Image


# Get the shape of the image
np.shape(Sample_Image)  


# In[ ]:


# Check out the type of the image
type(Sample_Image)


# In[ ]:


# Convert to numpy array
Sample_Image = tf.keras.preprocessing.image.img_to_array(Sample_Image)

# Sample_Image = np.array(Sample_Image)


# In[ ]:


# Confirm that the image is converted to Numpy array
type(Sample_Image)


# In[ ]:


# Obtain the max and min values
print('min pixel values = {}, max pixel values = {}'.format(Sample_Image.min(), Sample_Image.max()))

# Obtain the minimum and maximum pixel values for each channel separately
min_values = np.amin(sample_image_array, axis=(0, 1))
max_values = np.amax(sample_image_array, axis=(0, 1))

# Print the results
print('Minimum pixel values for each channel: {}'.format(min_values))
print('Maximum pixel values for each channel: {}'.format(max_values))


# In[9]:


# Normalize the input image
Sample_Image = np.array(Sample_Image)/248.0
sample_image_array = tf.keras.preprocessing.image.img_to_array(Sample_Image)
Sample_Image.shape


# In[ ]:


# Let's verify normalized images values!
print('min pixel values = {}, max pixel values = {}'.format(Sample_Image.min(), Sample_Image.max()))


# In[19]:


# Expanding Dimensions
Sample_Image = tf.expand_dims(Sample_Image, axis = 0)


# In[20]:


np.shape(Sample_Image)


# In[12]:


base_model.summary()


# In[21]:


# Maximize the activations of these layers

names = ['mixed3', 'mixed5', 'mixed7']


layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
deepdream_model = tf.keras.Model(inputs = base_model.input, outputs = layers)


# In[13]:


deepdream_model.summary()


# In[22]:


# Using Input and taking a look at the activations "Neuron outputs"
activations = deepdream_model(Sample_Image)
activations


# In[ ]:


len(activations)




# In[ ]:


x = tf.constant(2.0)


# In[ ]:


with tf.GradientTape() as g:
  g.watch(x)
  y = x * x * x
dy_dx = g.gradient(y, x) # Will compute to 12


# In[ ]:


dy_dx


# In[ ]:


x= tf.constant(5.0)




# In[ ]:



Sample_Image.shape


# In[23]:


Sample_Image = tf.squeeze(Sample_Image, axis = 0)


# In[ ]:


Sample_Image.shape


# In[24]:


def calc_loss(image, model):

  img_batch = tf.expand_dims(image, axis=0) # Convert into batch format
  layer_activations = model(img_batch) # Run the model
  print('ACTIVATION VALUES (LAYER OUTPUT) =\n', layer_activations)
  # print('ACTIVATION SHAPE =\n', np.shape(layer_activations))

  losses = [] # accumulator to hold all the losses
  for act in layer_activations:
    loss = tf.math.reduce_mean(act) # calculate mean of each activation 
    losses.append(loss)
  
  print('LOSSES (FROM MULTIPLE ACTIVATION LAYERS) = ', losses)
  print('LOSSES SHAPE (FROM MULTIPLE ACTIVATION LAYERS) = ', tf.shape(losses))
  print('SUM OF ALL LOSSES (FROM ALL SELECTED LAYERS)= ', tf.reduce_sum(losses))

  return  tf.reduce_sum(losses) # Calculate sum 


# In[25]:


loss = calc_loss(tf.Variable(Sample_Image), deepdream_model)


# In[ ]:


loss # Sum up the losses from both activations


@tf.function
def deepdream(model, image, step_size):
    with tf.GradientTape() as tape:
      # This needs gradients relative to `img`
      # `GradientTape` only watches `tf.Variable`s by default
      tape.watch(image)
      loss = calc_loss(image, model) # call the function that calculate the loss 
    gradients = tape.gradient(loss, image)

    print('GRADIENTS =\n', gradients)
    print('GRADIENTS SHAPE =\n', np.shape(gradients))

    # tf.math.reduce_std computes the standard deviation of elements across dimensions of a tensor
    gradients /= tf.math.reduce_std(gradients)  

   
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image


# In[27]:


def run_deep_dream_simple(model, image, steps = 2000, step_size = 0.01):
  # Convert from uint8 to the range expected by the model.
  image = tf.keras.applications.inception_v3.preprocess_input(image)

  for step in range(steps):
    loss, image = deepdream(model, image, step_size)
    
    if step % 100 == 0:
      plt.figure(figsize=(12,12))
      plt.imshow(deprocess(image))
      plt.show()
      print ("Step {}, loss {}".format(step, loss))

  # clear_output(wait=True)
  plt.figure(figsize=(12,12))
  plt.imshow(deprocess(image))
  plt.show()

  return deprocess(image)


# In[28]:


def deprocess(image):
  image = 255*(image + 1.0)/2.0
  return tf.cast(image, tf.uint8)


# In[ ]:


Sample_Image.shape


# In[ ]:


# Let's Load the image again and convert it to Numpy array 
Sample_Image = np.array(tf.keras.preprocessing.image.load_img('img_0.jpg'))
#dream_img = run_deep_dream_simple(model = deepdream_model, image = Sample_Image, steps = 2000, step_size = 0.001)




# In[ ]:


image = tf.keras.preprocessing.image.load_img("img_0.jpg")


# In[ ]:


plt.imshow(image)


# In[ ]:


# Name of the folder
dream_name = 'mars_eiffel'


# In[ ]:


# Blended image dimension

x_size = 910 # larger the image longer is going to take to fetch the frames 
y_size = 605


# In[ ]:


# Define Counters 
created_count = 0
max_count = 50


# In[ ]:


# This helper function loads an image and returns it as a numpy array of floating points

def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)


for i in range(0, 100):
    
     
    if os.path.isfile(r'C:\Users\Hashii\Desktop\{}\img_{}.jpg'.format(dream_name, i+1)):
        print("{} found, Fetching The Images ".format(i+1))
        
    else:
        # Call the load image funtion
        img_result = load_image(r'C:\Users\Hashii\Desktop\{}\img_{}.jpg'.format(dream_name, i))

    
        # Zoom the image 
        x_zoom = 2 # this indicates how quick the zoom is 
        y_zoom = 1
        
        # Chop off the edges of the image and resize the image back to the original shape. This gives the visual changes of a zoom
        img_result = img_result[0+x_zoom : y_size-y_zoom, 0+y_zoom : x_size-x_zoom]
        img_result = cv2.resize(img_result, (x_size, y_size))
        
        # Adjust the RGB value of the image
        img_result[:, :, 0] += 2  # red
        img_result[:, :, 1] += 2  # green
        img_result[:, :, 2] += 2  # blue
        
        # Deep dream model  
        img_result = run_deep_dream_simple(model = deepdream_model, image = img_result, steps = 500, step_size = 0.001)
        
        # Clip the image, convert the datatype of the array, and then convert to an actual image. 
        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        result = PIL.Image.fromarray(img_result, mode='RGB')
        
        # Save all the frames in the dream location
        result.save(r'C:\Users\Hashii\Desktop\{}\img_{}.jpg'.format(dream_name, i+1))
        
        created_count += 1
        if created_count > max_count:
            break



# # In[ ]:


# from google.colab import files
# uploaded = files.upload()


# # In[ ]:


# # Unzip the folder

# from zipfile import ZipFile
# file_name = "mars_eiffel.zip"

# with ZipFile(file_name, 'r') as zip:
#   zip.extractall()
#   print('Done')


# # In[ ]:


# # Path of all the frames

# dream_path = 'mars_eiffel'


# # In[ ]:


# # Define the codec and create VideoWriter object 
# # Download FFmeg 

# fourcc = cv2.VideoWriter_fourcc(*'XVID') # FourCC is a 4-byte code used to specify the video codec

# out = cv2.VideoWriter('deepdreamvideo.avi', fourcc , 5.0, (910, 605)) # Specify the fourCC, frames per second (fps),


# # In[ ]:


# for i in range(9999999999999):
    
#     # Get into the dream directory and looks for the number of images and then figure out what is the latest image. Hence with 
#     # this image we are going to start with and let it dream on and on
#     if os.path.isfile('mars_eiffel/img_{}.jpg'.format(i+1)):
#         pass
#     # Figure out how long the dream is 
#     else:
#         dream_length = i
#         break


# # In[ ]:


# dream_length


# # In[ ]:


# for i in range(dream_length):
    
#     # Build the frames of cv2.VideoWriter
#     img_path = os.path.join(dream_path,'img_{}.jpg'.format(i)) # join the dream path
    
#     print(img_path) # print the image path 
    
#     frame = cv2.imread(img_path)
#     out.write(frame)

# out.release()


