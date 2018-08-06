# combining data from different cities from the Bradbury data set
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

home_dir = '\\Users\\ThinkPad User\\Google Drive\\DeepSolaris'
os.chdir(home_dir)

Fneg = np.load('negatives_Fresno.npy')
Fpos = np.load('positives_Fresno.npy')
Sneg = np.load('negatives_Stockton.npy')
Spos = np.load('positives_Stockton.npy')
Mneg = np.load('negatives_Modesto.npy')
Mpos = np.load('positives_Modesto.npy')
Oneg = np.load('negatives_Oxnard.npy')
Opos = np.load('positives_Oxnard.npy')

Bradpos = np.concatenate((Fpos, Spos, Mpos, Opos), axis = 0)
positives_labels = np.full(19861, 1)

Bradneg = np.concatenate((Fneg, Sneg, Mneg, Oneg), axis = 0)
negatives_labels = np.full(19861, 0)

# freeing some memory
Fneg = []; del Fneg
Fpos = []; del Fpos
Sneg = []; del Sneg
Spos = []; del Spos
Mneg =[]; del Mneg
Mpos = []; del Mpos
Oneg = []; del Oneg
Opos = []; del Opos


### splitting up into train and test sets
train_imagesP, test_imagesP, train_labelsP, test_labelsP = train_test_split(Bradpos, positives_labels, test_size = 0.2, random_state = 0)
train_imagesN, test_imagesN, train_labelsN, test_labelsN = train_test_split(Bradneg, negatives_labels, test_size = 0.2, random_state = 0)

# freeing some memory
Bradpos = []; del Bradpos
Bradneg = []; del Bradneg
positives_labels = []; del positives_labels
negatives_labels = []; del negatives_labels

# creating one train and one test set
images_test = np.concatenate((test_imagesP, test_imagesN), axis = 0)
labels_test = np.concatenate((test_labelsP, test_labelsN))
images_train = np.concatenate((train_imagesP, train_imagesN), axis = 0)
labels_train = np.concatenate((train_labelsP, train_labelsN))
     
# freeing some memory
train_imagesP = []; del train_imagesP
train_imagesN = []; del train_imagesN
train_labelsP =[]; test_imagesN = []; del train_labelsP; del test_imagesN
train_labelsN = []; test_imagesP = []; test_labelsP = []; test_labelsN = []
del train_labelsN; del test_imagesP; del test_labelsP; del test_labelsN


# mixing them up
idxtest = np.random.choice(range(len(labels_test)), len(labels_test), replace = False)
idxtrain = np.random.choice(range(len(labels_train)), len(labels_train), replace = False)

images_test = images_test[idxtest, :, :, :]
labels_test = labels_test[idxtest]
images_train = images_train[idxtrain, :, :, :]
labels_train = labels_train[idxtrain]

# check if it is mixed well and labeled correctly
for i in range(30): 
    example = images_test[i,:,:,:]
    title = labels_test[i]
    example = cv2.resize(example, (300,300))
    name = 'Labels {}'.format(title)
    cv2.imshow(name, example)
    cv2.waitKey(1000)
cv2.destroyAllWindows()

for i in range(30): 
    example = images_train[i,:,:,:]
    title = labels_train[i]
    example = cv2.resize(example, (300,300))
    name = 'Labels {}'.format(title)
    cv2.imshow(name, example)
    cv2.waitKey(1000)
cv2.destroyAllWindows()


# saving the images
os.chdir(home_dir)
np.save('train_images_Bradbury', images_train)
np.save('train_labels_Bradbury', labels_train)
np.save('test_images_Bradbury', images_test)
np.save('test_labels_Bradbury', labels_test)


###############################################################################
#        Going on with the selected data set from Malof only for Fresno       #
###############################################################################

FnegTrain = np.load('negatives_Fresno_training.npy')
FnegLTrain = np.full(1823, 0)
FposTrain = np.load('positives_Fresno_training.npy')
FposLTrain = np.full(1823, 1)
FnegTest = np.load('negatives_Fresno_testing.npy')
FnegLTest = np.full(1022, 0)
FposTest = np.load('positives_Fresno_testing.npy')
FposLTest = np.full(1022, 1)

images_test = np.concatenate((FnegTest, FposTest), axis = 0)
labels_test = np.concatenate((FnegLTest, FposLTest))
images_train = np.concatenate((FnegTrain, FposTrain), axis = 0)
labels_train = np.concatenate((FnegLTrain, FposLTrain))

# randomly sample

idx = np.random.choice(range(len(labels_test)), len(labels_test), replace = False)
idxtrain = np.random.choice(range(len(labels_train)), len(labels_train), replace = False)

images_test = images_test[idx, :, :, :]
labels_test = labels_test[idx]
images_train = images_train[idxtrain, :, :, :]
labels_train = labels_train[idxtrain]

# check if it is mixed well and labeled correctly
for i in range(25): 
    example = images_test[i,:,:,:]
    title = labels_test[i]
    example = cv2.resize(example, (300,300))
    name = 'Labels {}'.format(title)
    cv2.imshow(name, example)
    cv2.waitKey(1200)
cv2.destroyAllWindows()

for i in range(30): 
    example = images_train[i,:,:,:]
    title = labels_train[i]
    example = cv2.resize(example, (300,300))
    name = 'Labels {}'.format(title)
    cv2.imshow(name, example)
    cv2.waitKey(1000)
cv2.destroyAllWindows()



# saving the images
os.chdir(home_dir)
np.save('train_images_Fresno', images_train)
np.save('train_labels_Fresno', labels_train)
np.save('test_images_Fresno', images_test)
np.save('test_labels_Fresno', labels_test)

train_images = np.load('train_images_Fresno.npy')
train_labels = np.load('train_labels_Fresno.npy')

import cv2
for i in range(30): 
    example = train_images[i,:,:,:]
    title = train_labels[i]
    example = cv2.resize(example, (300,300))
    name = 'Labels {}'.format(title)
    cv2.imshow(name, example)
    cv2.waitKey(500)
cv2.destroyAllWindows()

