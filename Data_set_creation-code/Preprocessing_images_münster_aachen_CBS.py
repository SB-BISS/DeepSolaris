# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:54:13 2018

@author: ThinkPad User
"""
    from keras.models import Sequential
    from keras.datasets import mnist
    import numpy as np
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.utils import np_utils
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    import os
    import re
    import cv2
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dropout
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    import cv2
    import numpy as np
    import os
    from time import time
    from datetime import timedelta
    import pandas as pd
    
    import math
    import matplotlib.pyplot as plt
    import pydot
    from keras.utils import plot_model
    from keras.utils.vis_utils import model_to_dot
    
    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
 
### setting the different directories  
    home_dir = '\\Users\\ThinkPad User\\Google Drive\\DeepSolaris'
    CBS_dir = str(home_dir + '\\CBS')
    m_dir = str(home_dir + '\\dop20_Münster')
    a_dir = str(home_dir + '\\dop20_Aachen')
    a_pos_dir = str(a_dir + '\\Positives_75x75')
    m_pos_dir = str(m_dir + '\\Positives_75x75')
    a_neg_dir = str(a_dir + '\\Negatives_75x75')
    m_neg_dir = str(m_dir + '\\Negatives_75x75')
    CBS_test_pos_dir = str(CBS_dir + '\\Test\\Positives')
    CBS_test_neg_dir = str(CBS_dir + '\\Test\\Negatives')
    CBS_train_pos_dir = str(CBS_dir + '\\Train\\Positives')
    CBS_train_neg_dir = str(CBS_dir + '\\Train\\Negatives')

    os.chdir(home_dir)

### reading in the images

    a_pos_images = [img for img in os.listdir(a_pos_dir)]
    a_neg_images = [img for img in os.listdir(a_neg_dir)]
    m_pos_images = [img for img in os.listdir(m_pos_dir)]
    m_neg_images = [img for img in os.listdir(m_neg_dir)]
    CBS_test_pos_images = [img for img in os.listdir(CBS_test_pos_dir)]
    CBS_train_pos_images = [img for img in os.listdir(CBS_train_pos_dir)]
    CBS_test_neg_images = [img for img in os.listdir(CBS_test_neg_dir)]
    CBS_train_neg_images = [img for img in os.listdir(CBS_train_neg_dir)]
    
    ### restricting the number of negative images to that of the positive images 
    ### to have a balanced data set
    a_neg_images = a_neg_images[0:len(os.listdir(a_pos_dir))]
    m_neg_images = m_neg_images[0:len(os.listdir(m_pos_dir))]
    CBS_test_neg_images = CBS_test_neg_images[0:len(CBS_test_pos_images)]
    CBS_train_neg_images = CBS_train_neg_images[0:len(CBS_train_pos_images)]

### bringing the images in shape and creating the output labels 
                # 1 -> positive
                # 0 -> negative
    max_h = 75
    max_w = 75    
   

### Helper functions

def pad_images(image, max_h = max_h, max_w = max_w):
    d, w, h = image.shape[::-1]
    top = round((max_h-h)/2)        # this way we center the picture
    bottom = max_h - h - top   # to be in the middle of the image
    left =  round((max_w-w)/2)
    right =  max_w - w - left
    padded_image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value= [0,0,0])
    padded_image = padded_image.reshape((1,max_h,max_w,3))
    return padded_image

def crop_images(image, height_out = max_h, width_out = max_w):
    images_cropped = np.ndarray((0, height_out, width_out, 3))
    height, width = image.shape[:2] 
    height_iter = (height // height_out) + 1 
    width_iter = (width // width_out) + 1    
    for i in range(1,width_iter):
       start_row = int((i-1) * width_out)
       end_row = int(i * width_out)
       for j in range(1,height_iter):
           start_col = int((j-1) * height_out)
           end_col = int(j * height_out)
           image_cropped = image[start_row:end_row , start_col:end_col]
           image_cropped = image_cropped.reshape((1,max_h,max_w,3))
           images_cropped = np.concatenate((images_cropped, image_cropped), axis = 0)
    return(images_cropped)


### loading the positives images into one array
    positives_complete = np.ndarray((0,max_h,max_w,3))
    positives_labels = np.empty(0,)
    deleted_p = []    
    img_num = len(a_pos_images) + len(m_pos_images) + len(CBS_test_pos_images) + len(CBS_train_pos_images)
    
    def load_positives(img_num = 10):
        global positives_complete
        global positives_labels       
        start_time = time()   
        
        for img_list in [a_pos_images, m_pos_images, CBS_test_pos_images, CBS_train_pos_images]:
            if(img_list == a_pos_images):
                os.chdir(a_pos_dir)
            elif(img_list == m_pos_images):
                os.chdir(m_pos_dir)
            elif(img_list == CBS_test_pos_images):
                os.chdir(CBS_test_pos_dir)
            else:
                os.chdir(CBS_train_pos_dir)
            for image_file in list(np.arange(0,len(img_list))): 
                if (img_list[image_file][-4:] != '.png'):
                    print('wrong file lies at index : {}'.format(image_file))
                else:
                    image = cv2.imread(img_list[image_file], cv2.IMREAD_COLOR)  # uint8 image
                    d, w, h = image.shape[::-1]
                    if (h > max_h or w > max_w):
                        deleted_p.append(image_file)
                    else:
                        norm_image = cv2.normalize(image, image, alpha=0, beta=0.99999, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        norm_image = pad_images(norm_image)
                        positives_complete = np.concatenate((positives_complete, norm_image))
                        output_label = 1
                        positives_labels = np.append(positives_labels, output_label)               
        end_time = time()
        time_dif = end_time - start_time
        print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))) + ' minutes')
        print('percentage of images lost due to size restriction: ' + str(round((len(deleted_p)/img_num)*100,2))+ '%')

    load_positives(img_num = img_num)
       
### loading the negative images

    negatives_complete = np.ndarray((0,max_h,max_w,3))
    negatives_labels = np.empty(0,)
    deleted_n = []    
    img_num_neg = len(a_neg_images) + len(m_neg_images) + len(CBS_test_neg_images) + len(CBS_train_neg_images)
    
    
    def load_negatives(img_num = 10):
        global negatives_complete
        global negatives_labels       
        start_time = time()   
        
        for img_list in [a_neg_images, m_neg_images, CBS_test_neg_images, CBS_train_neg_images]:
            if(img_list == a_neg_images):
                os.chdir(a_neg_dir)
            elif(img_list == CBS_test_neg_images):
                os.chdir(CBS_test_neg_dir)
            elif(img_list == CBS_train_neg_images):
                os.chdir(CBS_train_neg_dir)
            else:
                os.chdir(m_neg_dir)
            for image_file in list(np.arange(0,len(img_list))): 
                if (img_list[image_file][-4:] != '.png'):
                    print('wrong file lies at index : {}'.format(image_file))
                else:
                    image = cv2.imread(img_list[image_file], cv2.IMREAD_COLOR)  # uint8 image
                    d, w, h = image.shape[::-1]
                    if (h > max_h or w > max_w):
                        deleted_p.append(image_file)
                    else:
                        norm_image = cv2.normalize(image, image, alpha=0, beta=0.999999999999, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        norm_image = pad_images(norm_image)
                        negatives_complete = np.concatenate((negatives_complete, norm_image))
                        output_label = 0
                        negatives_labels = np.append(negatives_labels, output_label)               
        end_time = time()
        time_dif = end_time - start_time
        print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))) + ' minutes')
        print('percentage of images lost due to size restriction: ' + str(round((len(deleted_p)/img_num)*100,2))+ '%')

    load_negatives(img_num = img_num_neg)

### uniting positives and negatives in one array and saving those
    images_complete = np.ndarray((0, max_h, max_w, 3))
    labels_complete = np.empty(0, )
    images_complete = np.concatenate((positives_complete, negatives_complete), axis = 0)
    labels_complete = np.concatenate((positives_labels, negatives_labels))
     
# saving the images
    os.chdir(home_dir)
    np.save('images_Aachen_Münster_CBS_complete', images_complete)
    np.save('labels_Aachen_Münster_CBS_complete', labels_complete)
    
    

### displaying exemplary images
    example_pos = positives_complete[3,:,:,:]    
    example_pos = cv2.resize(example_pos, (0, 0), fx=6, fy=6)
    example_neg = negatives_complete[3,:,:,:]    
    example_neg = cv2.resize(example_neg, (0, 0), fx=6, fy=6)
    example_complete = images_complete[1,:,:,:]
    example_complete = cv2.resize(example_complete, (0,0), fx=6, fy=6)
    cv2.imshow('positive', example_pos)
    cv2.imshow('negative', example_neg)
    cv2.imshow('complete', example_complete)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
