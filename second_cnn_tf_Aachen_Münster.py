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
import time
from datetime import timedelta
import math
from sklearn.model_selection import train_test_split

from datetime import datetime



### setting the different directories  
home_dir = '\\Users\\ThinkPad User\\Google Drive\\DeepSolaris'
os.chdir(home_dir)


images_complete = np.load('images_Aachen_Münster_complete.npy')
labels_complete = np.load('labels_Aachen_Münster_complete.npy')
train_images, test_images, train_labels, test_labels = train_test_split(images_complete,labels_complete, test_size = 0.2, random_state = 0)
code_dir = '\\Users\\ThinkPad User\\Google Drive\\Master_thesis_H_Z_DL&NN\\Coding'
os.chdir(code_dir)


print("Size of:")
print("- Training-set:\t\t{}".format(len(train_labels)))
print("- Test-set:\t\t{}".format(len(test_labels)))


# We know that the images are 75 pixels in each dimension
img_size = 75
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 3
num_classes = 1


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
    
    for i in range(1, len(test_images)):
        print(test_images[i].max())
### Plot a few images to see if the data is correct
    images = test_images[9:18]
    cls_true = test_labels[9:18].astype(int)
    plot_images(images=images, cls_true=cls_true)    
    
    
    
    
###### Helper functions #######################################################

### new conv layer    
def new_conv_layer(input, num_input_channels, filter_size, num_filters, 
                   use_pooling=True, name = "conv"):  
    with tf.name_scope(name):    
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name = "W")
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]), name = "B") 
        # strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        layer = tf.nn.conv2d(input=input, filter=weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        layer += biases
        if use_pooling:
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME')
        layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations_after_pooling", layer)
        return layer, weights

#### flattening layer
def flatten_layer(layer, name = 'flatten'):
    with tf.name_scope(name):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

### fully-connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, name = "fc"):
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape =[num_inputs, num_outputs], stddev=0.05), name = "W")
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name = "B")
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations_after_pooling", layer)
        return layer    

### random batch helper function
train_batch_size = 64
def random_batch(train_batch_size = train_batch_size):
    num_images = len(train_images)
    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    # Use the random index to select random images and labels.
    x_batch = train_images[idx, :, :, :]
    x_batch = x_batch.astype(np.float32)
    y_batch = train_labels[idx]
    y_batch = y_batch.astype(np.float32)
    return x_batch, y_batch



###### CREATING THE MODEL ####################################################
LOGDIR = code_dir + '\\logs\\'

os.chdir(LOGDIR)
total_iterations = 0 


###  HYPERPARAMETERS 
learning_rate = 1e-4
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 32         # There are 36 of these filters.
# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

### creating a string-variable depending on the hyperparameter setting
def make_hparam_string(learning_rate = learning_rate, num_filters1 = num_filters1, num_filters2 = num_filters2, f_size1 = filter_size1, f_size2 = filter_size2):
  return "lr_%.0E_%s_%s_%s_%s" % (learning_rate, num_filters1, num_filters2, f_size1, f_size2)

hparam = make_hparam_string()


###model:
def scnd_model(num_iterations, learnrate = learning_rate, num_filters1 = num_filters1, num_filters2 = num_filters2, f_size1 = filter_size1, f_size2 = filter_size2, newGraph = False, param_str = hparam):
    tf.reset_default_graph()
    session = tf.Session()
    
    # placeholder variables
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name = 'x')    
    y_true = tf.placeholder(tf.float32, name='y_true')

    ### first conv layer
    layer_conv1, weights_conv1 = \
    new_conv_layer(input=x, num_input_channels=num_channels,
                   filter_size=f_size1, num_filters=num_filters1,
                   use_pooling=True, name = "conv1")
    ### second conv layer    
    layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
                   filter_size=f_size2, num_filters=num_filters2,
                   use_pooling=True, name = "conv2")
    ### flattening layer    
    layer_flat, num_features = flatten_layer(layer_conv2, name = 'flat')
    ### fully-connected layers 
    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,
                         num_outputs=fc_size, use_relu=True, name = "fc1")
    layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size,
                         num_outputs=num_classes, use_relu=False, name = "fc2")
    
    y_pred_c = tf.nn.sigmoid(layer_fc2)
    y_pred_c = tf.squeeze(y_pred_c, axis=1)
    y_pred = tf.round(y_pred_c)
     
    with tf.name_scope("cost"):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
        cost = tf.reduce_mean(tf.cast(cross_ent, tf.float32))
        tf.summary.scalar("cost", cost)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learnrate).minimize(cost)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        
    with tf.name_scope("statistics"):
        y_pred_avr = tf.reduce_mean(y_pred)
        tf.summary.scalar("prediction_averages", y_pred_avr)
        y_pred_c_avr = tf.reduce_mean(y_pred_c)
        tf.summary.scalar("pred_c_averages", y_pred_c_avr)
        y_pred_c_max = tf.reduce_max(y_pred_c)
        tf.summary.scalar("prediction_max", y_pred_c_max)
        y_pred_c_min = tf.reduce_min(y_pred_c)
        tf.summary.scalar("prediction_min", y_pred_c_min)
        y_pred_c_std = tf.sqrt(tf.reduce_mean(tf.square(y_pred_c - y_pred_c_avr)))
        tf.summary.scalar('standard_deviation', y_pred_c_std)

    # merging all tensorBoard summaries
    summ = tf.summary.merge_all()

    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    
    now = datetime.now()
    event_file_directory = logdir + now.strftime("%Y_%m_%d-%H-%M__") + str(hparam)
    if not os.path.exists(event_file_directory):
        os.makedirs(event_file_directory)
    writer = tf.summary.FileWriter(event_file_directory)
    if newGraph:
        writer.add_graph(session.graph)

    global total_iterations
    global accuracies
    global preds
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 5 == 0:
            [pred_std, pred_max, pred_cs, averages, c, s, acc] = session.run([y_pred_c_std, y_pred_c_max, y_pred_c_avr, y_pred_avr, cost, summ, accuracy], feed_dict=feed_dict_train)
            writer.add_summary(s, i)
            msg = "It{0:>6}, TrAcc{1:>6.1%}, Cost{2:>6.3}, mean {3:>6.2}, max {4:>6.2}, std: {5:>6.2}, avr: {6:>6.2}"
            print(msg.format(i + 1, acc, c, pred_cs, pred_max, pred_std, averages))
        if i % 500 == 0:
          saver.save(session, os.path.join(LOGDIR + now.strftime("%Y_%m_%d-%H-%M__") + str(hparam)))
        if i == (total_iterations + num_iterations - 1):
            [pred_examples, pred_c_examples] = session.run([y_pred, y_pred_c], feed_dict=feed_dict_train)
            print('predicition examples')
            print(pred_examples)
            print(pred_c_examples)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


#### GRIDSEARCH #########################################################

learning_vector = [1E-2, 1E-3, 1E-4]
filter_1_vector = [32]
filter_2_vector = [64]
fsize1_vector = [5]
fsize2_vector = [5]

# executing the model for different values of the hyperparameters
def main(num_iterations = 21, learning_vector = learning_vector):
  global hparam
  global total_iterations
  if (type(learning_vector) == int):
      learning_vector = generate_random_lr(num=learning_vector)
      print('generate random learning rates')
  else:
      print('use given learning rates')
  for learning_rate in learning_vector:
    for num_filter1 in filter_1_vector:
      for num_filter2 in filter_2_vector:
          for filter_size1 in fsize1_vector: 
              for filter_size2 in fsize2_vector:
                if (num_filter2 < num_filter1):
                    print('number of filters should increase')
                else:
                    total_iterations = 0 
                    lr = round(learning_rate, 5)
                    hparam = "lr_%s_%s_%s_%s_%s" % (lr, num_filter1, num_filter2, filter_size1, filter_size2)
                    print('Starting run for %s' % hparam)   
                    scnd_model(num_iterations = num_iterations, learnrate = learning_rate, num_filters1 = num_filter1, num_filters2 = num_filter2, f_size1 = filter_size1, f_size2 = filter_size2, newGraph = False, param_str = hparam)
  print('Done training!')
  print('Run `tensorboard --logdir=trainingx:logs` to see the results.')


### implement random search for hyperparamters
lr_min = 1E-6
lr_max = 1E-2
def generate_random_lr(lr_min = lr_min, lr_max = lr_max, num = 1):
    '''generate random learning rate'''
    global random_learning_rate
    random_learning_rate = []
    for i in range(num):
        r_l_r = np.random.uniform(lr_min, lr_max)
        random_learning_rate.append(r_l_r)
    return random_learning_rate

def make_hparam_string(learning_rate = learning_rate, num_filters1 = num_filters1, num_filters2 = num_filters2, f_size1 = filter_size1, f_size2 = filter_size2):
  return "lr_%.0E_%s_%s_%s_%s" % (learning_rate, num_filters1, num_filters2, f_size1, f_size2)




# checking accuracy before any training: 
session.run(tf.global_variables_initializer())

print_test_accuracy(show_confusion_matrix= True, show_example_errors= True) 
 # Accuracy on Test-Set: 83.3% (209 / 251)
 # the confusion matrix shows that the CNN predicts everything to not
 # be a solar panel

scnd_model(11) 

session.run(tf.global_variables_initializer())
print_test_accuracy()
#83.3% -> always predicts no sp
optimize(201)





###### HELPER FUNCTIONS

   
### Example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = test_images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = test_labels[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

### Confusion matrix
def plot_confusion_matrix(cls_pred, 
                          normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cls_true = test_labels
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    classes = ['no sp', 'sp']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
    
### performance accuracy    
test_batch_size = 64

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False, 
                        show_first_100_preds = True):
    num_test = len(test_images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = test_images[i:j, :]
        labels = test_labels[i:j]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred, feed_dict=feed_dict)
        i = j
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
        plot_confusion_matrix(cls_pred=cls_pred)
    # Show the first hundred predictions made:
    if show_first_100_preds:
        print("First 100 predictions:")
        print(cls_pred)
        
### Helper function for conv weights
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
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
    
### Helper function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
### Input Images  
def plot_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()