import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split


# Loading + training,validation test split
cars0 = glob.glob('./vehicles/GTI_Far/*.png')
cars1 = glob.glob('./vehicles/GTI_MiddleClose/*.png')
cars2 = glob.glob('./vehicles/GTI_Left/*.png')
cars3 = glob.glob('./vehicles/GTI_Right/*.png')
cars4 = glob.glob('./vehicles/KITTI_extracted/*.png')
#cars4 += glob.glob('./vehicles/augmentation/*.png')
notcars1 = glob.glob('./non-vehicles/Extras/*.png')
notcars1 += glob.glob('./non-vehicles/GTI/*.png')

# split 70% training 20% validation 10% test set
frac1 = 0.7
l0,l1,l2,l3,l4,l5=len(cars0),len(cars1),len(cars2),len(cars3),len(cars4),len(notcars1)
L1 = (frac1*np.array([l0,l1,l2,l3,l4,l5])).astype('int')
frac2 = 0.9
l0,l1,l2,l3,l4,l5=len(cars0),len(cars1),len(cars2),len(cars3),len(cars4),len(notcars1)
L2 = (frac2*np.array([l0,l1,l2,l3,l4,l5])).astype('int')

cars_train = cars0[:L1[0]] + cars1[:L1[1]] + cars2[:L1[2]] + cars3[:L1[3]] + cars4[:L1[4]]
notcars_train = notcars1[:L1[5]]

cars_val = cars0[L1[0]:L2[0]] + cars1[L1[1]:L2[1]] + cars2[L1[2]:L2[2]] + cars3[L1[3]:L2[3]] + cars4[L1[4]:L2[4]]
notcars_val = notcars1[L1[5]:L2[5]]

cars_test = cars0[L2[0]:] + cars1[L2[1]:] + cars2[L2[2]:] + cars3[L2[3]:] + cars4[L2[4]:]
notcars_test = notcars1[L2[5]:]


print('Number of samples in cars training set: ', len(cars_train))
print('Number of samples in notcars training set: ', len(notcars_train))

print('Number of samples in cars validation set: ', len(cars_val))
print('Number of samples in notcars validation set: ', len(notcars_val))

print('Number of samples in cars test set: ',len(cars_test))
print('Number of samples in notcars test set: ',len(notcars_test))


# Save the data for easy access
pickle_file = 'data.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'cars_train': cars_train,
                'notcars_train': notcars_train,
                'cars_val': cars_val,
                'notcars_val': notcars_val,
                'cars_test': cars_test,
                'notcars_test': notcars_test
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')


data_file = 'data.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
cars_train = data['cars_train']
notcars_train = data['notcars_train']
cars_val = data['cars_val']
notcars_val = data['notcars_val']
cars_test = data['cars_test']
notcars_test = data['notcars_test']

i=22
a_car = plt.imread(cars_train[i])
not_a_car = plt.imread(notcars_train[i])
cars_train[i],notcars_train[i]

font_size=30
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(a_car)
ax1.set_title('a car', fontsize=font_size)
ax2.imshow(not_a_car)
ax2.set_title('not a car', fontsize=font_size)
plt.rc('xtick', labelsize=font_size) 
plt.rc('ytick', labelsize=font_size) 
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(a_car)
ax1.set_title('a car', fontsize=font_size)
ax2.imshow(not_a_car)
ax2.set_title('not a car', fontsize=font_size)
plt.rc('xtick', labelsize=font_size) 
plt.rc('ytick', labelsize=font_size) 
plt.savefig('./images/car_notcar.png')


