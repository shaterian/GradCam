# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 02:39:42 2020

@author: sshateri
"""
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
import math 

import os 
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

from vis.utils import utils 

import json
from keras.preprocessing.image import load_img, img_to_array

from skimage.transform import resize

from video_capture import VideoCaptureYUV
import cv2

# Node look up
from node_lookup import NodeLookup


MODEL_PATH = './inception_model'

# YUV Path
TCM_MAIN_PATH = 'G:\\jpeg_data\\TCM_Y_only\\yuv'

# JPEG Path
JPEG_MAIN_PATH  = 'G:\\jpeg_data\\validation_original'
JPEG_PATH = 'G:\\jpeg_data\\SEP_JPEG\\yuv'
PATH_TO_EXCEL_TCM = 'D:\\NEW_TCM_EXPERIMENTS\\tcm_data\\TCM_inference_10K.xlsx'
PATH_TO_EXCEL_JPEG = 'D:\\NEW_TCM_EXPERIMENTS\\jpg_data\\JPEG_inference10K.xlsx'
EXAMPLE_PATH = 'D:\\NEW_TCM_EXPERIMENTS\\visualize_inception_featureMaps-master\\gradcamp_examples'
MAIN_Y = 'H:\\TCM_JPEG_Y_only\\yuv'

#PATH_TO_EXCEL = 'TCM_inference' + "_" + str(START) + "_" + str(END) + '.xls'


def sep_get_image_data(imgID, QP, isTCM = True):
	# Parse the YUV and convert it into RGB
	# this function return data of image
	original_img_ID = imgID
	imgID = str(imgID).zfill(8)
	shard_num  = round(original_img_ID/10000);
	folder_num = math.ceil(original_img_ID/1000);
	current_jpg = JPEG_MAIN_PATH + '\\' + str(folder_num) + '\\' + 'ILSVRC2012_val_' + imgID + '.JPEG'
	print(current_jpg)
	jpg_img = cv2.imread (current_jpg)
	size = jpg_img.shape
	height = size [0]
	width = size [1]
    
	if isTCM:
		real_height =  int ( math.ceil(height/8)*8 ) 
		real_width = int (math.ceil(width/8)*8 )
		padded_size = ( real_height , real_width )   
		print("padded_size" , padded_size)
		current_image = TCM_MAIN_PATH + '\\' + str(folder_num) + '\\'+ str(QP)+ '\\' + 'ILSVRC2012_val_' + imgID + '_' + str(QP) + '.yuv'  
		print(current_image)       
		videoObj = VideoCaptureYUV(current_image , padded_size, isGrayScale=True)        
		ret, yuv, rgb = videoObj.getYUVAndRGB()
		image_data = rgb         
		image_data = rgb[0:height , 0:width]
		figure_title =  'TCM_' + 'ILSVRC2012_val_' + imgID + '_' + str(QP) + '.yuv' 
	else:
		if QP ==  110:
			size = (height, width)
			current_image = MAIN_Y + '\\' + str(folder_num) + '\\'+ 'ILSVRC2012_val_' + imgID  + '.yuv'  
			print(current_image)       
			videoObj = VideoCaptureYUV(current_image , size, isGrayScale=True)        
			ret, yuv, rgb = videoObj.getYUVAndRGB()
			image_data = rgb         
			image_data = rgb[0:height , 0:width]
			figure_title =  'ORG_' + 'ILSVRC2012_val_' + imgID + '_' + str(QP) + '.yuv' 	
		

		else:
			size = (height, width)
			output_path = JPEG_PATH
			current_image = output_path + '\\' + str(folder_num) + '\\' +'ILSVRC2012_val_' + imgID + '-QF-' + str(QP) + '.yuv'
			videoObj = VideoCaptureYUV(current_image , size, isGrayScale=True)
			ret, yuv, rgb = videoObj.getYUVAndRGB()
			image_data = rgb  
			figure_title =  'JPG_' + 'ILSVRC2012_val_' + imgID + '-QF-' + str(QP) + '.yuv'
		
	return image_data, figure_title



def run_predictions(model, img, QP, idx, sheet, style, gt_label_list):

    y_pred            = model.predict(img[np.newaxis,...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
#    topNclass         = 5
        ## grad cam
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]

    # ID --> English string label.
   
    # Current_rank = -1
    current_rank = -1

    for rank, node_id in enumerate(class_idxs_sorted):
        human_string = classlabel[node_id]
        score = y_pred[node_id] 
        
        print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))

        if(gt_label_list[idx] == human_string):
            
            print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))
            # Write the rank and the score
            #print(gt_label_list[idx])

            # Set the current rank (rank starts from 0)
            current_rank = 1 + rank
                       
            # Stop looping once you find it in the rank
            break

    return current_rank

def readAndPredictLoopOptimized():
    
    # Create the excel sheet workbook
    path_to_excel = PATH_TO_EXCEL
    rb = xlrd.open_workbook(path_to_excel, formatting_info=True)
    wb = copy(rb)
    sheet = wb.get_sheet(0)
    style = XFStyle()
    style.num_format_str = 'general'

    print('Done excel sheet creation')

    # Get the ground truth list
    sheet_r       = rb.sheets()[0]
    gt_label_list = sheet_r.col_values(1)

    # Construct QP list to get the full range
    qp_list = construct_qp_list() # 12 qp

    t = 0 
    ## remove it 
    
    for imgID in range(START,END):
        startTime = time.time()

        original_img_ID = imgID

        actual_idx = original_img_ID
        if (actual_idx == START):         
            sess = tf.Session()
            #sess = tf.Session()
            create_graph()
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            print('New session group has been created')
        else :

	        if (actual_idx - 1) % 200 == 0:
	            config = tf.ConfigProto(device_count = {'GPU': 0})
	            sess = tf.Session()
	            #sess = tf.Session()
	            create_graph()
	            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
	            print('New session group has been created')

        if (actual_idx == 4703 or actual_idx == 520 ):
            continue
        else:
            imgID = str(imgID).zfill(8)

            folder_num = math.ceil(original_img_ID/1000)
            glob_path_jpg = JPEG_MAIN_PATH + '\\' + str(folder_num) + '\\' + 'ILSVRC2012_val_' + imgID + '*.JPEG'
            
            filesList = glob.glob(glob_path_jpg)
            current_jpg_image = filesList[0]
            
            jpg_img = cv2.imread (current_jpg_image)

            size = jpg_img.shape
            height = size [0]
            width = size [1]
            print("real size", size)
            for qp_idx, QP in enumerate(qp_list):
                
                real_height =  int ( math.ceil(height/8)*8 ) 
                real_width = int (math.ceil(width/8)*8 )
                padded_size = ( real_height , real_width )   
                print("padded_size" , padded_size)
                current_image = TCM_MAIN_PATH + '\\' + str(folder_num) + '\\'+ str(QP)+ '\\' + 'ILSVRC2012_val_' + imgID + '_' + str(QP) + '.yuv'
                
                videoObj = VideoCaptureYUV(current_image , padded_size, isGrayScale=True)
                
                ret, yuv, rgb = videoObj.getYUVAndRGB()
                image_data = rgb 
                
                image_data = rgb[0:height , 0:width]
                print("image_data shape", image_data.shape)
                current_rank = run_predictions(sess, image_data, softmax_tensor, QP, actual_idx , sheet, style, gt_label_list, qp_idx)
                # displayRGB(image_data)
                print(current_rank)

            if (actual_idx) % 200 == 0:
                tf.reset_default_graph()
                sess.close()
            
    #
            t += time.time() - startTime
            if not original_img_ID % 10 :
                print ('image %d is done in %f seconds' % (original_img_ID, t))
                t = 0
            # if not original_img_ID % 10000:
            #     wb.save(path_to_excel)
            # # print('Final Save...')
            wb.save(path_to_excel)






                           
def plot_map(grads, img):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    plt.suptitle("Pr(class={}) = {:5.2f}".format(
                      classlabel[class_idx],
                      y_pred[0,class_idx]))

def pre_process(img_name):
    
    _img = load_img("creative_commons_elephant.jpg",target_size=(224,224))
    img               = img_to_array(_img)
    img               = preprocess_input(img)
    return img




def sep_pre_process(image_data):
    
    img               = preprocess_input(image_data.astype('float64'))
    img = resize(image_data, (244,244))
    return img



def gradCam (model , class_idx, img) :
    
        
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'predictions')

    # Swap softmax with linear
    model.layers[layer_idx].activation = keras.activations.linear
    model = utils.apply_modifications(model)

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
    
#    plt.imshow(Lc_Grad_CAM)
#    plt.show()

    return grad_CAM 

def visualize(model, img, figure_title):
    
    y_pred            = model.predict(img[np.newaxis,...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
    topNclass         = 5
    for i, idx in enumerate(class_idxs_sorted[:topNclass]):
        print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
        i + 1,classlabel[idx],idx,y_pred[0,idx]))
    
        ## grad cam
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]

    ## select class of interest
    class_idx           = class_idxs_sorted[0]
    grad_CAM = gradCam (model, class_idx, img) 

    plot_map(grad_CAM,img) 
    plt.suptitle(figure_title, fontsize=16)
    
    if not os.path.exists(os.path.join(EXAMPLE_PATH ,str(imgID))) :
        os.mkdir(os.path.join(EXAMPLE_PATH ,str(imgID)))

    plt.savefig(os.path.join(EXAMPLE_PATH ,str(imgID) , figure_title.split('.')[0]+  '.JPEG' ), dpi=600)

#
if __name__ == '__main__':
    
    model = InceptionV3(weights='imagenet',include_top=True)
    model.summary()
    for ilayer, layer in enumerate(model.layers):
        print("{:3.0f} {:10}".format(ilayer, layer.name))

    CLASS_INDEX = json.load(open("imagenet_class_index.json"))
    classlabel  = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)][1])
    print("N of class={}".format(len(classlabel)))
        
    model = InceptionV3(weights='imagenet',include_top=True)
    image_name =  r"D:\NEW_TCM_EXPERIMENTS\visualize_inception_featureMaps-master\gradCam\creative_commons_elephant.jpg"
    img = pre_process(image_name)
    y_pred            = model.predict(img[np.newaxis,...])
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]
    topNclass         = 5
    for i, idx in enumerate(class_idxs_sorted[:topNclass]):
        print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
        i + 1,classlabel[idx],idx,y_pred[0,idx]))
    
        ## grad cam
    class_idxs_sorted = np.argsort(y_pred.flatten())[::-1]

    ## select class of interest
    class_idx           = class_idxs_sorted[0]
    grad_CAM = gradCam (model, class_idx, img) 

    plot_map(grad_CAM,img)




#if __name__ == '__main__':
#    
#    model = InceptionV3(weights='imagenet',include_top=True)
##    model.summary()
##    for ilayer, layer in enumerate(model.layers):
##        print("{:3.0f} {:10}".format(ilayer, layer.name))
#
#    CLASS_INDEX = json.load(open("imagenet_class_index.json"))
#    classlabel  = []
#    for i_dict in range(len(CLASS_INDEX)):
#        classlabel.append(CLASS_INDEX[str(i_dict)][1])
#    print("N of class={}".format(len(classlabel)))
#    
#    QP = 28
#    imgID = 1608
#    image_data,figure_title =  sep_get_image_data(imgID, QP, isTCM = True) 
#    im =  sep_pre_process(image_data)
#    
#        
#    
#    visualize(model, im, figure_title)
#
#    QP = 15
#    imgID = 1608
#    image_data,figure_title =  sep_get_image_data(imgID, QP, isTCM = False) 
#    im = sep_pre_process(image_data)
#
#    visualize(model, im, figure_title)