import numpy as np
import cv2
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
import datetime as dt
import os
import random


def split_img(img, n, k, thold, index=0):

    '''
    :param img: cv2 image
    :param n: number of subimages to split recursively (in 1 dim)
    :param k: steps of recursion
    :param thold: for keeping edges
    :param ind: for saving images name
    :return: none, saves images
    '''

    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #skip saving blank images
    if cv2.countNonZero(gs_img) == 0:
        return

    size = tuple(img.shape[:2])
    gs_img = cv2.equalizeHist(gs_img)
    gs_img = cv2.GaussianBlur(gs_img, (3, 3), 0, 0)
    gs_img = cv2.medianBlur(gs_img, 11)

    edges = cv2.Canny(gs_img, 250, 255)

    pic = edges
    sq = n
    ind = (len(pic) // sq)
    sub = []
    subim = []

    for i in range(sq):
        for j in range(sq):
            sub.append(pic[i*ind:(i + 1) * ind, j*ind:(j + 1) * ind])



    means = [(i.mean()) for i in sub]

    means = np.array(means).reshape(-1, sq)
    means = np.nan_to_num(means)
    means = Normalizer(norm='max').fit_transform(means)

    means = cv2.resize(means, size , interpolation=0)
    for i in range(size[0]):
        for j in range(size[1]):
            if means[i, j] < thold:
                img[i, j] = np.array([0, 0, 0])

    for i in range(sq):
        for j in range(sq):
            subim.append(img[i*ind:(i + 1) * ind, j*ind:(j + 1) * ind])

    if k == 0:
        if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 0.1*img.shape[0]*img.shape[1]:
            return

        #use color filter and morph operators to isolate roofs
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (10, 0, 0), (100, 255, 255))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        mask = cv2.medianBlur(mask, 15, 15)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)

        if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 0.1*img.shape[0]*img.shape[1]:
            return

        fname = 'split/'+(dt.datetime.now()).strftime('%s%f')+'.png'
        cv2.imwrite(fname,img)
        print ('Wrote IMG: '+fname)
        return

    else:
        print ('Processing ' + str(index)+str(k))
        for i in range(len(subim)):
                split_img(subim[i], n, k-1, thold-0.1, index=i)

def cut_img (path, size):

    '''
    :param path: folder with split images
    :param size: size of final images to get
    :return: none, writes images
    '''

    for file in os.listdir(path):
        if file.endswith(".png"):

            pic = cv2.imread(path+file)
            sq = (len(pic) // size)
            ind = size
            sub = []

            for i in range(sq):
                for j in range(sq):
                    im = pic[i * ind:(i + 1) * ind, j * ind:(j + 1) * ind]
                    if cv2.countNonZero(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) > 0.1 * im.shape[0] * im.shape[1]:
                        sub.append(im)

            for i in sub:
                fname = path + 'cut/' + (dt.datetime.now()).strftime('%s%f') + '.png'
                cv2.imwrite(fname, i)



#testing, pick 10 random pictures and process them
dir = os.listdir('data/')
files = [dir[random.randrange(len(dir))] for i in range(10)]


for file in files:
    if file.endswith(".png"):
        print ('Processing '+ file)
        im = cv2.imread('data/'+file)
        split_img(im, 10, 1, 0.5)

print ('Done splitting, now cutting')

cut_img('split/', 50)

print ('Done cutting')



