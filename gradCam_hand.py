import keras.backend as K
from scipy.ndimage.interpolation import zoom
import keras
import tensorflow as tf
import vis ## keras-vis
import matplotlib.pyplot as plt
import numpy as np
print("keras      {}".format(keras.__version__))
print("tensorflow {}".format(tf.__version__))
#
#from keras.applications.vgg16 import VGG16, preprocess_input
#model = VGG16(weights='imagenet')

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
model = InceptionV3(weights='imagenet',include_top=True)

model.summary()

for ilayer, layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))

import json
CLASS_INDEX = json.load(open("imagenet_class_index.json"))
classlabel  = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
print("N of class={}".format(len(classlabel)))

from keras.preprocessing.image import load_img, img_to_array
#_img = load_img("duck.jpg",target_size=(224,224))
_img = load_img("creative_commons_elephant.jpg",target_size=(224,224))
plt.imshow(_img)
plt.show()

img               = img_to_array(_img)
img               = preprocess_input(img)
y_pred            = model.predict(img[np.newaxis,...])
class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
topNclass         = 5


for i, idx in enumerate(class_idxs_sorted[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,y_pred[0,idx]))


from vis.utils import utils         
# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

## grad cam
class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]

## select class of interest
class_idx           = class_idxs_sorted[0]
## feature map from the final convolusional layer

final_fmap_index    = utils.find_layer_idx(model, 'mixed10')
penultimate_output  = model.layers[final_fmap_index].output

## define derivative d loss^c / d A^k,k =1,...,512
layer_input          = model.input
## This model must already use linear activation for the final layer
loss                 = model.layers[layer_idx].output[...,class_idx]
grad_wrt_fmap        = K.gradients(loss,penultimate_output)[0]

## create function that evaluate the gradient for a given input
# This function accept numpy array
grad_wrt_fmap_fn     = K.function([layer_input,K.learning_phase()],
                                  [penultimate_output,grad_wrt_fmap])

## evaluate the derivative_fn
fmap_eval, grad_wrt_fmap_eval = grad_wrt_fmap_fn([img[np.newaxis,...],0])

# For numerical stability. Very small grad values along with small penultimate_output_value can cause
# w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())

print(grad_wrt_fmap_eval.shape)
alpha_k_c           = grad_wrt_fmap_eval.mean(axis=(0,1,2)).reshape((1,1,1,-1))
Lc_Grad_CAM         = np.maximum(np.sum(fmap_eval*alpha_k_c,axis=-1),0).squeeze()

## upsampling the class activation map to th esize of ht input image
scale_factor        = np.array(img.shape[:-1])/np.array(Lc_Grad_CAM.shape)
_grad_CAM           = zoom(Lc_Grad_CAM,scale_factor)
## normalize to range between 0 and 1
arr_min, arr_max    = np.min(_grad_CAM), np.max(_grad_CAM)
grad_CAM            = (_grad_CAM - arr_min) / (arr_max - arr_min + K.epsilon())

plt.imshow(Lc_Grad_CAM)
plt.show()

plt.figure(figsize=(20,5))
plt.plot(alpha_k_c.flatten())
plt.xlabel("Feature Map at Final Convolusional Layer")
plt.ylabel("alpha_k^c")
plt.title("The {}th feature map has the largest weight alpha^k_c".format(
    np.argmax(alpha_k_c.flatten())))
plt.show()


                           
def plot_map(grads):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(_img)
    axes[1].imshow(_img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    plt.suptitle("Pr(class={}) = {:5.2f}".format(
                      classlabel[class_idx],
                      y_pred[0,class_idx]))

plot_map(grad_CAM)
