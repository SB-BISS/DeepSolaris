import imutils
import cv2
import os

def rotate(folder_name):

    prit = 0
    for file in os.listdir(folder_name):
        if file.endswith('png') or file.endswith('jpg'):
            img = cv2.imread(folder_name+file)

            for i in range(4):
                img = imutils.rotate(img, angle=90)
                cv2.imwrite(folder_name+str(i)+file, img)

        prit += 1
        if prit %100 == 0:
            print ('Processing ', prit)

def mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 0, 0), (100, 255, 255))

    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    mask = cv2.medianBlur(mask, 11)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    mask = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(img, img, mask=mask)

    return img