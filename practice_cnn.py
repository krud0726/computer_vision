#!/usr/bin/env python
# coding: utf-8

# # import package

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras as k
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.datasets import cifar10


# # 1.  Load Datasets & Normalization

# In[20]:


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[21]:


# check the number and configuration of train data.
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# check the number and configuration of test data.
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# In[22]:


"""
[0, 3] : 0 이상 3 이하
[0, 3) : 0 이상 3 미만
(0, 5] : 0 초과 5 이하
(0, 5) : 0 초과 5 미만
""" 
# declare labels
labels = ["airplane",'automobile', 'bird','cat',
          'deer','dog','frog','horse','ship','truck']

# create an index array for random samples
index = np.random.randint(len(X_train), size = 100)

# 100 random data extraction from training data
random_sample = X_train[index, :,:,:]

# I use subplots to print 100 random data.
fig, ax = plt.subplots(10,10, figsize = (16,16))

# adjust spacing and height of subplot
plt.subplots_adjust(wspace = 0.4, hspace = 0.2)

# to print images easily, make the axes object array into a one-dimensional array.
ax = ax.ravel()

# declare an index array of labels for each of the 100 images.
label_index = y_train[index]

# print 100 images and each label.
for i in range(100):
    ax[i].imshow(random_sample[i])
    # set title and turn off axes and labels
    ax[i].set_title(labels[label_index[i][0]])
    ax[i].axis('off')
    


# In[23]:


# normalize the data to values between 0 and 1.
X_train = X_train / 255.0
X_test = X_test / 255.0

# check the result
print(X_train)
print(X_test)


# # 2. Training Model

# In[24]:


# labels of the data are transformed into a one hot encoding using the 'to_categorical' function.
y_train_encoding = to_categorical(y_train, 10)
y_test_encoding = to_categorical(y_test,10)
print(y_train_encoding)


# In[25]:


# construct the CNN model with the following structure.
model = Sequential()

# output data size (10 categories)
num_classes = 10

# batch size, epoch, validation_split 
batch_size = 64
epoch = 10
validation_split = 0.2

# Conv2D Extract features using 32 filters. The size of the input data is defined in the first layer.
# Calculate the features of the image using filters in the convolution layer.
# Activate the calculation result using the activation function.
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(64, 5, activation='relu'))

# sub-sampling the result of the convolution calculation. Extracts the largest value within the specified filter size.
model.add(MaxPool2D())

model.add(Conv2D(128, 5, activation='relu'))
model.add(MaxPool2D())

# normalize using dropout technique.
model.add(Dropout(0.25))

# The image is converted to one dimension through the flatten() process.
model.add(Flatten())

# configure a fully connected layer. A fully connected layer means a layer that is connected to all neurons in the previous layer.
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))

# It constitutes an output layer with 10 units.
model.add(Dense(num_classes, activation='softmax'))


"""
compile is a method related to model optimization. A method that tells the model what to do for optimization

* loss: Defines the loss calculation function. The reason the loss calculation function is important is that it serves as an index for judging the model result.

* optimizer: Defines the optimization function. That is, it helps to find the lowest loss.

* metrics : It means the value used to evaluate the model.

What the compile method does is to define an optimization method that minimizes loss.

"""
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Check the structure of the created model.
model.summary()

# train the model through the given options in the PDF.
history = model.fit(X_train, y_train_encoding, 
          batch_size = batch_size, 
          epochs = epoch,
         validation_split=validation_split)


# # 3. Evaluate & Predict

# In[26]:


# After fitting, the history object is received, and the data is converted into a dataframe of pandas for convenient use.
df = pd.DataFrame(history.history)

# check the data
df


# In[27]:


# set figure size
plt.figure(figsize = (12,4))

# Compare the accuracy of the train set and the validation set and output it through a plot.
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.legend()

# Compare the loss of the train set and the validation set and output it through a plot.
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'] , label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()

# print the result
plt.show()


# In[28]:


# use the predict_classes function to predict the X_test data.
predict_class = model.predict_classes(X_test)

# check out the result
print(predict_class)
print(predict_class.shape)
print(len(predict_class))


# In[29]:


# create an index array for random samples
random_index = np.random.randint(len(predict_class), size = 12)

# 12 random data extraction from test data
random_test_sample = X_test[random_index, :,:,:]

# I use subplots to print 12 random data.
fig, ax = plt.subplots(3,4, figsize=(16,10))

# to print images easily, make the axes object array into a one-dimensional array.
ax = ax.ravel()

# declare an index array of labels for each of the 12 images.
test_label_index = y_test[random_index]

# declare an index array of predict labels for each of the 12 images.
predict_label_index = predict_class[random_index]

# print 12 images and each label and predicted label
for i in range(12):
    # print out the image
    ax[i].imshow(random_test_sample[i])
    
    # show the prediction and the actual value(label)
    ax[i].text(0,0, 'Prediction: %s' % labels[predict_label_index[i]], color = 'k', backgroundcolor='w', )
    ax[i].text(0,4, 'LABEL: %s' % labels[test_label_index[i][0]], color = 'k', backgroundcolor='w', )
    
    # turn off axes and labels
    ax[i].axis('off')

