import numpy as np
import cv2
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt


def split_img(img, n, k, thold, index=0, name='cut0'):

    '''
    :param img: cv2 image
    :param n: number of subimages to split recursively
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

    names = []
    for i in range(sq):
        for j in range(sq):
            subim.append(img[i*ind:(i + 1) * ind, j*ind:(j + 1) * ind])
            names.append(str(i*ind)+ 'x' +str((i + 1) * ind)+',' + str (j*ind) +'x'+ str((j + 1) * ind))

    if k == 0:
        if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) == 0:
            return

        #use color filter and morph operators to isolate roofs
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (10, 0, 0), (100, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))


        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)

        fname = name+'.png'
        cv2.imwrite(fname,img)
        print ('Wrote IMG: '+fname)
        return

    else:
        print ('Processing ' + str(index)+str(k))
        for i in range(len(subim)):
                split_img(subim[i], n, k-1, thold-0.1, index=i, name =str(index)+names[i])


im = cv2.imread('data/0D_dop20rgbi_32296_5621_1_nw_result.png')
#im = cv2.imread('test.png')
split_img(im, 4, 2, 0.5)