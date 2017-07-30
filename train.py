import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
import cv2

training_file = "training_data/train.p"
testing_file = "test_data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)

n_classes = (max(y_train) - min(y_train)) + 1

count_train = np.zeros(n_classes)
count_test = np.zeros(n_classes)

print('count train length', len(count_train))

for i in range(n_train):
    idx = int(y_train[i])
    count_train[idx] +=1

for i in range(n_test):
    idx = int(y_test[i])
    count_test[idx] +=1


# grouping for counting class numbers
class_groups = np.zeros(3)
for i in range(n_classes):
    if (count_train[i] < 1000):
        class_groups[0] += 1
    elif (count_train[i] > 1000 and count_train[i] < 2000):
        class_groups[1] += 1
    else:
        class_groups[2] += 1


# upper and lower limit of each class_group
limits = np.zeros([n_classes, 2])
limits[0, 0] = 0
limits[0, 1] = count_train[0]

for i in range(1, n_classes):
    limits[i, 0] = limits[i - 1, 1] + 1
    limits[i, 1] = limits[i, 0] + count_train[i] - 1


### normalization

# rgb to gray
def to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale

def grayscale_array(X_Arr):
    X_out = np.zeros((len(X_Arr),32,32,1))
    for i in range(len(X_Arr)):
        img = X_Arr[i].squeeze()
        X_out[i,:,:,0] = to_grayscale(img)
    return X_out


# normalizing sample number per class_groups
X_out = X_train
y_out = y_train

for n in range(n_classes): #For each class
    if (count_train[n] < 1000):
        #Add twice
        cl_start = int(limits[n, 0])
        cl_end = int(limits[n, 1])
        X_swap1 =  X_train[cl_start:cl_end,:,:,:]
        X_swap2 =  X_train[cl_start:cl_end,:,:,:]
        y_d1 = y_train[cl_start:cl_end]
        y_d2 = y_train[cl_start:cl_end]

        X_out = np.concatenate([X_out, X_swap1, X_swap2])
        y_out = np.concatenate([y_out, y_d1, y_d2])

    elif (count_train[n] < 2000):
        cl_start = int(limits[n, 0])
        cl_end = int(limits[n, 1])
        X_swap =  X_train[cl_start:cl_end,:,:,:]
        y_d = y_train[cl_start:cl_end]

        X_out = np.concatenate([X_out, X_swap])
        y_out = np.concatenate([y_out, y_d])

    else:
        cl_start = int(limits[n, 0])
        cl_end = int((limits[n, 1] + limits[n, 0]) / 2)
        X_swap =  X_train[cl_start:cl_end,:,:,:]
        y_d = y_train[cl_start:cl_end]

        X_out = np.concatenate([X_out, X_swap])
        y_out = np.concatenate([y_out, y_d])


X_train_gray = grayscale_array(X_out)
X_train_norm= X_train_gray / 255


# shuffling training and test sets
from sklearn.utils import shuffle

X_train_shuffled, y_train_shuffled = shuffle(X_train_norm, y_out, random_state = 3)
X_test_shuffled, y_test_suffled = shuffle(X_test, y_test, random_state = 3)

#split  training/validation/
n_train = len(X_train_shuffled)
valid_range = int(n_train*0.8)

X_train_ini = (X_train_shuffled[0:valid_range + 1])
X_valid_ini = (X_train_shuffled[valid_range + 1:n_train + 1])
y_train_ini = (y_train_shuffled[0:valid_range + 1])
y_valid_ini = (y_train_shuffled[valid_range + 1:n_train + 1])

print(X_train_ini.shape)
print(X_valid_ini.shape)









