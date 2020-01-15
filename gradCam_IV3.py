# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:56:21 2020

@author: sshateri
"""


from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from vis.utils import utils
from keras.preprocessing import image
import numpy as np
from keras import activations
from matplotlib import pyplot as plt

from vis.visualization import visualize_cam,overlay


model = InceptionV3(weights='imagenet',include_top=True)

# Utility to search for layer index by name
layer_idx = utils.find_layer_idx(model,'predictions')

#swap with softmax with linear classifier for the reasons mentioned above
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


from vis.utils import utils
from matplotlib import pyplot as plt


plt.rcParams['figure.figsize']=(18,6)

img1 = utils.load_img('creative_commons_elephant.jpg',target_size=(299,299))
print(img1.shape)
print(type(img1))
print(type(img1[0][0][0]))
plt.imshow(img1)
plt.show()


preprocess_img = preprocess_input(img1.astype('float64'))
print(preprocess_img.shape)
print(type(preprocess_img[0][0][1]))
plt.imshow(preprocess_img)
plt.show()

model.summary()

penultimater_layer_idx = utils.find_layer_idx(model,'mixed10')
print(penultimater_layer_idx)


heatmap = visualize_cam(model, layer_idx, filter_indices=292, #20 for ouzel and 292 for tiger 
                                seed_input=preprocess_img, backprop_modifier=None, # relu and guided don't work
                        penultimate_layer_idx = 310 #310 is concatenation before global average pooling
                       )
plt.imshow(heatmap)
plt.show()


plt.imshow(overlay(img1, heatmap))
plt.show()