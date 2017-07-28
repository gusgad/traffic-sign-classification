import pickle
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np

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

print(class_groups)
