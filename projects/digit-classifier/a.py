from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))
plt.show()


import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_ordering('th')

np.random.seed(7)

X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test=X_train.reshape(X_test.shape[0],1,28,28).astype('float32')

X_train=X_train/255
X_test=X_test/255

#one hot encodeing

# Output=[0,0,0,0,0,1,0]

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

num_classed=y_train.shape[1]


















