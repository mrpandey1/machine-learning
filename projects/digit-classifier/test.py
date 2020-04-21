# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:23:18 2020

@author: Baraka
"""

from PIL import Image
from keras.datasets import mnist
import numpy as np

(X_train,y_train),(X_test,y_test)=mnist.load_data()
for i in np.random.randint(0,10000+1,10):
    arr2im=Image.fromarray(X_train[i])
    arr2im.save('{}.png'.format(i),"PNG")