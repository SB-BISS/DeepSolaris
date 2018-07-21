import cv2
import numpy as np
import pandas as pd
import cyvlfeat as vlf
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.externals import joblib

#given a folder iterate over all images and extract features in a numpy array
def get_sift_descriptors(folder_name, step=8, ws=16):

    '''
    :param folder_name: folder of images
    :return: list of 128 features for each image in the folder
    '''

    kp = None
    j = 0
    for file in os.listdir(folder_name):
        if file.endswith('png'):

            #image is read and coverted to grayscale
            img = cv2.imread(folder_name+file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #the dense sift function from cyvlfeat is used
            #step and window size are baalnced to get enough features without getting too many
            if kp is None:
                kp = (vlf.sift.dsift(gray, step=step, window_size=ws)[1])
                continue

            kp = np.concatenate((kp, (vlf.sift.dsift(gray, step=step, window_size=ws)[1])))
        j+=1
        if j%50 == 0:
            print ('Processing ', j)

    return kp



#kpoints = get_sift_descriptors('data/airplanes_train/')
#print (kpoints.shape)

def get_features(img, kmeans):

    '''
    :param img: image to extract features from
    :param kmeans: kmeans model used to extract words
    :return: histogram of image features
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = (vlf.sift.dsift(gray, step=8, window_size=16)[1])

    preds = []
    for line in kp:
        preds.append(kmeans.predict(line.reshape(1,-1)))

    return np.histogram(np.array(preds),kmeans.n_clusters)[0]


def full_extraction(data_arr, k_words, load_kmeans=False, km_jobs=4):

    '''
    :param data_arr: data to extract features from as numpy array n*128
    :param k_words: number of words (clusters) to use
    :param load_kmeans: if already have kmeans model for this word number, load it
    :param km_jobs: number of parallel kmeans to run
    :return: train and test X,Y vectors to train and test classifiers
    '''

    if not load_kmeans:
        print ('Training kmeans, this will take a while')
        kmeans = KMeans(k_words, max_iter=300, verbose=1, precompute_distances=True, n_init=km_jobs, n_jobs=km_jobs)
        kmeans.fit(data_arr)
        joblib.dump(kmeans, 'saved/'+str(k_words)+'_words.pkl')
        print ('KMeans Model Dumped to File')
    else:
        kmeans = joblib.load('saved/'+str(k_words)+'_words.pkl')

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    print ('Starting to load features')
    for file in os.listdir('data/airplanes_train/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/airplanes_train/' + file)
            X_train.append(get_features(img, kmeans))
            Y_train.append(0)

    print ('Loaded airplanes train')
    for file in os.listdir('data/airplanes_test/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/airplanes_test/' + file)
            X_test.append(get_features(img, kmeans))
            Y_test.append(0)

    print ('Loaded airplanes test')
    for file in os.listdir('data/cars_train/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/cars_train/' + file)
            X_train.append(get_features(img, kmeans))
            Y_train.append(1)

    print ('Loaded cars train')
    for file in os.listdir('data/cars_test/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/cars_test/' + file)
            X_test.append(get_features(img, kmeans))
            Y_test.append(1)
    print ('Loaded cars test')

    for file in os.listdir('data/faces_train/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/faces_train/' + file)
            X_train.append(get_features(img, kmeans))
            Y_train.append(2)
    print ('Loaded faces train')

    for file in os.listdir('data/faces_test/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/faces_test/' + file)
            X_test.append(get_features(img, kmeans))
            Y_test.append(2)
    print ('Loaded faces test')

    for file in os.listdir('data/motorbikes_train/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/motorbikes_train/' + file)
            X_train.append(get_features(img, kmeans))
            Y_train.append(3)
    print ('Loaded mbikes train')

    for file in os.listdir('data/motorbikes_test/'):
        if file.endswith('jpg'):
            img = cv2.imread('data/motorbikes_test/' + file)
            X_test.append(get_features(img, kmeans))
            Y_test.append(3)
    print ('Loaded mbikes test')

    X_train = normalize(np.array(X_train), norm='max')
    X_test = normalize(np.array(X_test), norm='max')
    Y_train = np.array(Y_train).reshape(-1, )
    Y_test = np.array(Y_test).reshape(-1, )

    np.savetxt('saved/xtrain'+str(k_words)+'.txt', X_train, delimiter=',')
    np.savetxt('saved/xtest'+str(k_words)+'.txt', X_test, delimiter=',')
    np.savetxt('saved/ytrain'+str(k_words)+'.txt', Y_train, delimiter=',')
    np.savetxt('saved/ytest'+str(k_words)+'.txt', Y_test, delimiter=',')

    print ('Finished loading features')
    return X_train,X_test,Y_train,Y_test