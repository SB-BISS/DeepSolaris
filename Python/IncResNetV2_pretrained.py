# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:53:11 2018

@author: Hannah
"""
import os
import itertools

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from time import time
from datetime import timedelta
import math
from sklearn.model_selection import train_test_split
from datetime import datetime


from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

no_of_epochs = 0




### Helper metrics functions
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
    
    
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)














### check if data are in the right format
K.image_data_format() == 'channels_last'

### setting the different directories  
home_dir = '\\Users\\ThinkPad User\\Google Drive\\DeepSolaris'
code_dir = '\\Users\\ThinkPad User\\Google Drive\\Master_thesis_H_Z_DL&NN\\Coding'
log_dir = code_dir + '\\logs'
os.chdir(home_dir)
images_complete = np.load('images_Aachen_Münster_CBS_complete.npy')
labels_complete = np.load('labels_Aachen_Münster_CBS_complete.npy')
train_images, test_images, train_labels, test_labels = train_test_split(images_complete,labels_complete, test_size = 0.2, random_state = 0)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train_labels)))
print("- Test-set:\t\t{}".format(len(test_labels)))


### create the base pre-trained model
IncResNet_base = InceptionResNetV2(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = IncResNet_base.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)
IncResNet_model = Model(inputs=IncResNet_base.input, outputs=preds)
for layer in IncResNet_base.layers:   # only train the new layers first
    layer.trainable = True
IncResNet_model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
                        metrics=['accuracy', precision, recall, fmeasure])

### visualizing the IncResNet_model DOESNT WORK
os.chdir(home_dir)
tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, 
                         write_graph=False, write_images=True)

###################### first training stage ###################################
def train_IncResNet_model(epochs = 21):
    start_time = time()   
    global no_of_epochs
    IncResNet_model.fit(x = train_images, y = train_labels, epochs = epochs, 
                        callbacks=[tbCallBack])
    no_of_epochs = no_of_epochs + epochs
    end_time = time()
    time_dif = end_time - start_time
    print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))) + ' minutes')

train_IncResNet_model(epochs = 30)    

result = IncResNet_model.evaluate(test_images, test_labels)
for name, value in zip(IncResNet_model.metrics_names, result):
    print(name, value)



##################### second training stage ###################################
for i, layer in enumerate(IncResNet_base.layers):
   if ('mix' in layer.name):
       print(i, layer.name)
for layer in IncResNet_model.layers[:275]:
   layer.trainable = False
for layer in IncResNet_model.layers[275:]:
   layer.trainable = True

IncResNet_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='binary_crossentropy', metrics = ['acc'])
def train2_IncResNet_model(epochs = 21):
    global no_of_epochs
    IncResNet_model.fit(x = train_images, y = train_labels, epochs = epochs)
    no_of_epochs = no_of_epochs + epochs
train2_IncResNet_model(epochs = 21)

result = IncResNet_model.evaluate(test_images, test_labels)
for name, value in zip(IncResNet_model.metrics_names, result):
    print(name, value)

### making predictions on the test data
predictions = IncResNet_model.predict(test_images).squeeze(axis = 1)
binary_preds = predictions.round()
print(predictions[0:10], binary_preds[0:10])

# Plot a few images to see if the data is correct
images = test_images[9:18]
cls_true = test_labels[9:18].astype(int)
cls_pred = binary_preds[9:18]
plot_images(images=images, cls_true=cls_true, cls_pred = cls_pred)    

### checking test accuracy, example errors and confusion matrix
print_test_accuracy

### checking which layers are convolutional layers
for i, layer in enumerate(IncResNet_base.layers):
   if ('conv' in layer.name):
       print(i, layer.name)

plot_weights(input_channel = 0, layer_index = 1)
plot_weights(input_channel = 1, layer_index = 1)



### saving the IncResNet_model
now = datetime.now()
name =  'IncResNetV2_' + str(no_of_epochs) + now.strftime("__%Y_%m_%d-%H-%M")
os.chdir(code_dir)
IncResNet_model.save(name)















#*****************************************************************************
################# HELPER FUNCTIONS ###########################################
#
#*****************************************************************************

### Helper function for confusion matrix #####################################
import itertools
classes = ['no sp', 'sp']

def plot_confusion_matrix(cm = cm, classes = classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum()
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")     
    plt.tight_layout(pad = 2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


###### Helper function for plotting images ####################################
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show() 
    
### Example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = test_images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = test_labels[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def print_test_accuracy(show_example_errors=True,
                        show_confusion_matrix=True, 
                        show_first_10_preds = True):
    num_test = len(test_images)
    #cls_pred = np.zeros(shape=num_test, dtype=np.int)
    cls_pred = IncResNet_model.predict(test_images).squeeze(axis = 1).round()
    cls_true = test_labels
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        cm = confusion_matrix(test_labels, cls_pred.round())
        plot_confusion_matrix(cm = cm)
    # Show the first hundred predictions made:
    if show_first_10_preds:
        print("First 10 predictions:")
        print(cls_pred[0:10])


#### Helper function for plotting the output of a convolutional layer
#def plot_conv_layer(layer, image):
#    feed_dict = {x: [image]}
#    values = IncResNet_model.run(layer, feed_dict=feed_dict)
#    num_filters = values.shape[3]
#    num_grids = math.ceil(math.sqrt(num_filters))
#    fig, axes = plt.subplots(num_grids, num_grids)
#    for i, ax in enumerate(axes.flat):
#        if i<num_filters:
#            img = values[0, :, :, i]
#            ax.imshow(img, interpolation='nearest', cmap='binary')
#        ax.set_xticks([])
#        ax.set_yticks([])
#    plt.show()

### Helper function for conv weights
def plot_weights(input_channel=0, layer_index = 1):
    w = IncResNet_model.layers[layer_index].get_weights()[0]
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()















   
    
    
    
    
    
    
    
    
    
    
    
