import os
home_dir = 'C:\\Users\\ThinkPad User\\Google Drive\\Master_thesis_H_Z_DL&NN\\Coding\\'
image_dir = 'E:\\CaliforniaImages\\'
save_dir = 'C:\\Users\\ThinkPad User\\Google Drive\\DeepSolaris\\'

os.chdir(home_dir)

import json
import numpy as np
import cv2
from random import randint

# size transformer (from 0.3 resolution to 0.25 resolution)
trf = 0.3/0.2
 # size of the adapted image
trfimg = round(trf * 5000)


# this method takes a json file
def read_json(file):
    with open(file) as json_file: # open the json file
        return json.load(json_file) # return the json object
    
# this method creates a dictionary with the filename as a key and the central
# polygon coordinates as values

def create_polygonset_from_image_name2(data):
    filename_collection = {} # initialize
    for polygons in data["polygons"]: # for each polygon entry in json
        # remove this if you want every folder processed
        if polygons["city"] in "Stockton": # filter for city
            # remove this if you want every filename processed
            #if polygons["image_name"] in "10sfg735670": # filter for filename
                key = "./" + polygons["city"] + "/" + polygons["image_name"] + ".tif" #declare the key for the dictionary
                if key in filename_collection: # check if key is already in dictionary
                    #for pol in polygons["centroid_latitude_pixels"]: # add new polygons to existing ones
                    lat = polygons["centroid_latitude_pixels"]
                    long = polygons["centroid_longitude_pixels"]
                    filename_collection[key].append([long, lat])
                else: # if not create new entry
                    polygon_collection = [] # initialize collection
                    lat = polygons["centroid_latitude_pixels"]
                    long = polygons["centroid_longitude_pixels"]
                    polygon_collection.append([long,lat])
                    filename_collection[key] = polygon_collection
#                    for pol in polygons["polygon_vertices_pixels"]: # add polygons to collection
#                        polygon_collection.append(pol)
#                    filename_collection[key] = polygon_collection # create new entry in dictionary
    return filename_collection # return dictionary



# this method takes a filename with polygons
positives_Stockton = np.ndarray((0, 75, 75, 3), dtype = 'uint8')
def cut_image_and_label(file, polygons):
    global positives_Stockton
    im = cv2.imread(file) # open image
    im = cv2.normalize(im, im, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, 
                       dtype=cv2.CV_32F)
    im = cv2.resize(im, (trfimg,trfimg))
    w, h, d = im.shape # get the image size
    for polygon in polygons: 
        c1 = int(polygon[0])
        c2 = int(polygon[1])
        c1 = round(c1*trf)
        c2 = round(c2*trf)
        c1range = randint(10, 64)
        c2range = randint(10, 64)
        ll1 = c1-c1range
        ul1 = ll1+75
        ll2 = c2-c2range
        ul2 = ll2+75      
        if(ll1<0):
            ll1 = 0
            ul1 = 75
        if(ul1>7500):
            ll1 = 7425
            ul1 = 7500
        if(ll2<0):
            ll2 = 0
            ul2 = 75
        if(ul2>7500):
            ll2 = 7425
            ul2 = 7500
        cropped_img = im[ll2:ul2, ll1:ul1] # crop file
        cropped_img = cropped_img.reshape((1,75,75,3))
        positives_Stockton = np.concatenate((positives_Stockton, cropped_img))
        #cv2.rectangle(showcase,(ll2,ul1),(ul2,ll1),(80,255,120),15)
#        print('Img: {}, n°: {}, coordinates: {}'.format(name, counter, polygon))
#        counter += 1 # increase the counter with each polygon point
        


data = read_json(home_dir + "\\SolarArrayPolygons.json") # read the json file

# get back the polygons per image
polygons_per_image = create_polygonset_from_image_name2(data) 
for key in polygons_per_image.keys():
  print(key)
print(len(polygons_per_image.keys()))



###############################################################################
#                       NOW THE NEGATIVES                                     #
###############################################################################
  
negatives_Stockton = np.ndarray((0, 75, 75, 3), dtype = 'uint8')
l1 = randint(38, 112)
l2 = randint(38, 112)
def cut_image_and_label_negatives(file, coordinates):
    global negatives_Stockton
    global showcase
    img = cv2.imread(file) # open image
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, 
                       dtype=cv2.CV_32F)
    img = cv2.resize(img, (trfimg,trfimg))
#    showcase = img
    for center1 in range(l1, 4970, 500):
        for center2 in range(l2, 4970, 500):
            overlap = 0
            for polygon in coordinates: 
                if polygon[0]== '_NaN_' or polygon[0]== '_NaN_':
                    break
                c1 = int(polygon[0])
                c2 = int(polygon[1])
                # if the new image center is in the range of one of the solar arrays, then break the loop
                if (c1-76 < center1 < c1+76):
                    overlap = 1
                    break
                elif (c2-76 < center2 < c2+76):
                    overlap = 1
                    break
            if (overlap ==0):
                ll1 = round((center1*trf))-37
                ul1 = round((center1*trf))+38
                ll2 = round((center2*trf))-37
                ul2 = round((center2*trf))+38
                cropped_img = img[ll2:ul2, ll1:ul1] # crop file
                cropped_img = cropped_img.reshape((1,75,75,3))
                negatives_Stockton = np.concatenate((negatives_Stockton, cropped_img))
                #cv2.rectangle(showcase,(ll2,ul1),(ul2,ll1),(80,0,255),15)
                name = str(file).split("\\")[-1]

# unittest cut_image_and_label_negatives:
    # if there are no polygons there should be 4.356 negatives created
#coordinates = []
#cut_image_and_label_negatives(newimg, coordinates)
    # unit test confirmed: exactly 4356 images
    
# 2nd unittest, to see on image if rectangles do not overlap: 
expl = next(iter(polygons_per_image))
newimg = image_dir + str(expl).split('/')[1] + '\\' + str(expl).split('/')[2]
exampleimg = cv2.imread(newimg)
exo = cv2.resize(exampleimg, (500,500))
exr = cv2.resize(exampleimg, (750,750))

cv2.imshow('orignial', exo)
cv2.imshow('resized', exr)
cv2.waitKey()
cv2.destroyAllWindows()

cut_image_and_label_negatives(newimg, polygons_per_image[expl]) # crop the image
cut_image_and_label(newimg, polygons_per_image[expl]) # crop the image

visualtest = showcase
visualtest = cv2.resize(visualtest, (1000,1000))
cv2.imshow('visual test', visualtest)
cv2.waitKey()
cv2.destroyAllWindows()
### it works !!!

# creating all the negative subimages
counter = 0
negatives_Stockton = np.ndarray((0, 75, 75, 3), dtype = 'uint8')
for img in polygons_per_image: # loop through the image names
    newimg = image_dir + str(img).split('/')[1] + '\\' + str(img).split('/')[2]
    cut_image_and_label_negatives(newimg, polygons_per_image[img]) # crop the image
    counter += 1
    print('Img: {}, n°: {}'.format(newimg, counter))
    
# check if there are really no solar panels
for i in range(30): 
    example = negatives_Stockton[i,:,:,:]
    example = cv2.resize(example, (300,300))
    name = 'Picture {}'.format(i)
    cv2.imshow(name, example)
    cv2.waitKey(900)
cv2.destroyAllWindows()

for i in range(2000, 2010): 
    example = negatives_Stockton[i,:,:,:]
    example = cv2.resize(example, (300,300))
    name = 'Picture {}'.format(i)
    cv2.imshow(name, example)
    cv2.waitKey(1200)
cv2.destroyAllWindows()


#creating all the positive subimages
counter = 0
positives_Stockton = np.ndarray((0, 75, 75, 3), dtype = 'uint8')
for img in polygons_per_image: # loop through the image names   
    newimg = image_dir + str(img).split('/')[1] + '\\' + str(img).split('/')[2]
    cut_image_and_label(newimg, polygons_per_image[img]) # crop the image
    counter += 1 # increase the counter with each polygon point
    print('Img: {}, n°: {}'.format(newimg, counter))


# check images if there really are solar panels
for i in range(30): 
    example = positives_Stockton[i,:,:,:]
    example = cv2.resize(example, (300,300))
    name = 'Picture {}'.format(i)
    cv2.imshow(name, example)
    cv2.waitKey(900)
cv2.destroyAllWindows()

for i in range(2000, 2030): 
    example = positives_Stockton[i,:,:,:]
    example = cv2.resize(example, (300,300))
    name = 'Picture {}'.format(i)
    cv2.imshow(name, example)
    cv2.waitKey(900)
cv2.destroyAllWindows()


fn = (save_dir + 'positives_Stockton')
np.save(fn , positives_Stockton)    


### now saving the negative examples 

fn = (save_dir + 'negatives_Stockton')
index = np.random.choice(range(len(negatives_Stockton)), len(positives_Stockton), replace=False)
final_negatives_Stockton = negatives_Stockton[index, :, :, :]
np.save(fn , final_negatives_Stockton)
    
