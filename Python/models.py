from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense, Layer
from keras.layers import concatenate, Input, dot, Lambda
from keras import backend as K
import tensorflow as tf
import numpy as np

def small_vgg(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def vgg(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def vgg_fourier_mid(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1

    input1 = Input(inputShape)
    input2 = Input((width, height, 1))
   
    c1_1 = Conv2D(32, (3, 3), padding="same", activation='relu')(input1)
    bn1_1 = BatchNormalization(axis=chanDim)(c1_1)
    mp1_1 = MaxPooling2D(pool_size=(3, 3))(bn1_1)
    do1_1 = Dropout(0.25)(mp1_1)
    c1_2 = Conv2D(32, (3, 3), padding="same", activation='relu')(input2)
    bn1_2 = BatchNormalization(axis=chanDim)(c1_2)
    mp1_2 = MaxPooling2D(pool_size=(3, 3))(bn1_2)
    do1_2 = Dropout(0.25)(mp1_2)
            
            
    c2_1 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_1)
    bn2_1 = BatchNormalization(axis=chanDim)(c2_1)
    c2__1 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_1)
    bn2__1 = BatchNormalization(axis=chanDim)(c2__1)
    mp2_1 = MaxPooling2D(pool_size=(2, 2))(bn2__1)
    do2_1 = Dropout(0.25)(mp2_1)
    c2_2 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_2)
    bn2_2 = BatchNormalization(axis=chanDim)(c2_2)
    c2__2 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_2)
    bn2__2 = BatchNormalization(axis=chanDim)(c2__2)
    mp2_2 = MaxPooling2D(pool_size=(2, 2))(bn2__2)
    do2_2 = Dropout(0.25)(mp2_2)
          
    merge = concatenate([do2_1,do2_2])

    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(merge)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(c3)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c3 = Dropout(0.25)(c3)

    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c3)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c4)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c4 = Dropout(0.25)(c4)

    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c4)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c5)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = MaxPooling2D(pool_size=(2, 2))(c5)
    c5 = Dropout(0.25)(c5)

    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c5)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c6)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = MaxPooling2D(pool_size=(2, 2))(c6)
    c6 = Dropout(0.25)(c6)

    out = Flatten()(c6)
    out = Dense(1024, activation='relu')(out)
    out = BatchNormalization()(out)
    out =  Dropout(0.5)(out)


    out = Dense(classes, activation='softmax')(out)

    model = Model(inputs=[input1,input2], outputs=out)
    # return the constructed network architecture
    return model


def vgg_fourier_end(width, height, depth, classes, ft_shape=(75,75)):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1

    input1 = Input(inputShape)
    input2 = Input((*ft_shape, 1))

    c1_1 = Conv2D(32, (3, 3), padding="same", activation='relu')(input1)
    bn1_1 = BatchNormalization(axis=chanDim)(c1_1)
    mp1_1 = MaxPooling2D(pool_size=(3, 3))(bn1_1)
    do1_1 = Dropout(0.25)(mp1_1)

    c1_2 = Conv2D(1, ft_shape, padding="same", activation='relu')(input2)
    do1_2 = Dropout(0.3)(c1_2)
    fl_2 = Flatten()(do1_2)
    dn_2 = Dense(1024, activation='relu')(fl_2)


    c2_1 = Conv2D(64, (3, 3), padding="same", activation='relu')(do1_1)
    bn2_1 = BatchNormalization(axis=chanDim)(c2_1)
    c2__1 = Conv2D(64, (3, 3), padding="same", activation='relu')(bn2_1)
    bn2__1 = BatchNormalization(axis=chanDim)(c2__1)
    mp2_1 = MaxPooling2D(pool_size=(2, 2))(bn2__1)
    do2_1 = Dropout(0.25)(mp2_1)


    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(do2_1)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = Conv2D(64, (3, 3), padding="same", activation='relu')(c3)
    c3 = BatchNormalization(axis=chanDim)(c3)
    c3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c3 = Dropout(0.25)(c3)

    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c3)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = Conv2D(128, (3, 3), padding="same", activation='relu')(c4)
    c4 = BatchNormalization(axis=chanDim)(c4)
    c4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c4 = Dropout(0.25)(c4)

    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c4)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = Conv2D(256, (3, 3), padding="same", activation='relu')(c5)
    c5 = BatchNormalization(axis=chanDim)(c5)
    c5 = MaxPooling2D(pool_size=(2, 2))(c5)
    c5 = Dropout(0.25)(c5)

    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c5)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = Conv2D(512, (3, 3), padding="same", activation='relu')(c6)
    c6 = BatchNormalization(axis=chanDim)(c6)
    c6 = MaxPooling2D(pool_size=(2, 2))(c6)
    c6 = Dropout(0.25)(c6)

    fl_1 = Flatten()(c6)
    dn_1 = Dense(1024, activation='relu')(fl_1)

    merge = concatenate([dn_1,dn_2])
    out = Dense(1024, activation='relu')(merge)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)

    out = Dense(classes, activation='softmax')(out)

    model = Model(inputs=[input1, input2], outputs=out)
    # return the constructed network architecture
    return model

def vgg2(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1


    model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


def large_vgg(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def add_top(model, classes):
    chanDim = -1

    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(1024, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
def add_fs_top(model, classes, size=8192):
    model.add(Dense(size))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

def add_top_small(model, classes):
    chanDim = -1

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

class FFT_Filter(Layer):

    def __init__(self, **kwargs):
        super(FFT_Filter, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True
                                      )
        super(FFT_Filter, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        
        ftimage = np.fft.fft2(x.eval())
        ftimage = np.fft.fftshift(ftimage)
        fft = tf.cast(ftimage,dtype=tf.complex64)
        real = tf.real(fft)
        imag = tf.imag(fft)
        fil = real*self.kernel
        full = tf.complex(fil, imag)
        return abs(tf.ifft2d(full))

    def compute_output_shape(self, input_shape):
        return input_shape

class FFT_IN(Layer):

    def __init__(self, **kwargs):
        super(FFT_IN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True
                                      )
        super(FFT_IN, self).build(input_shape)  # Be sure to call this somewhere!

  
    def call(self, x):
        fft = tf.fft2d(tf.cast(x, dtype=tf.complex64))
        real = tf.real(fft)*tf.sigmoid(self.kernel) #because you want to cut frequencies, not enhance them
        imag = tf.imag(fft)
        full = tf.complex(real,imag)
        full = abs(tf.ifft2d(full))
        return full

    def compute_output_shape(self, input_shape):
        return input_shape

class FFT_OUT(Layer):

    def __init__(self, **kwargs):
        super(FFT_OUT, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True
                                      )
        super(FFT_OUT, self).build(input_shape)  # Be sure to call this somewhere!

    def my_ifft(self,x):
        myx= np.fft.ifft2(x)
        return myx
    
        
        
    def call(self, x):
        real = tf.real(x)*tf.sigmoid(self.kernel)
        imag = tf.imag(x)
        #fil = real
        full = tf.complex(real, imag)
        #exit = tf.py_func(self.my_ifft,[full],tf.complex64)
        myvar = abs(tf.ifft2d(full))
        #myvar=exit
        return myvar #tf.cast(exit,dtype=tf.float32)*self.kernel
        #return #exit 

    def compute_output_shape(self, input_shape):
        return input_shape


def fft_filter_clf(classes, shape=(75,75)):

    model = Sequential()

    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))
    #model.add(Dropout(0.25))
    #model.add(FFT_OUT(name='FFT_OUT'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    return model


def fft_vgg(classes, shape=(75,75)):
    
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model = Sequential()

    model.add(FFT_IN(input_shape=(*shape,3),name='FFT_IN'))


   
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model



if __name__ == '__main__':

    model = fft_filter_clf(2)
    model.summary()