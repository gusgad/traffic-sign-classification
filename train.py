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
X_test_gray = grayscale_array(X_test)

X_train_norm= X_train_gray / 255
X_test_norm= X_test_gray / 255


# shuffling training and test sets
from sklearn.utils import shuffle

X_train_shuffled, y_train_shuffled = shuffle(X_train_norm, y_out, random_state = 3)
X_test_shuffled, y_test_shuffled = shuffle(X_test_norm, y_test, random_state = 3)

#split  training/validation/
n_train = len(X_train_shuffled)
valid_range = int(n_train*0.8)

X_train_ini = (X_train_shuffled[0:valid_range + 1])
X_valid_ini = (X_train_shuffled[valid_range + 1:n_train + 1])
y_train_ini = (y_train_shuffled[0:valid_range + 1])
y_valid_ini = (y_train_shuffled[valid_range + 1:n_train + 1])


# NN model
import tensorflow as tf

from tensorflow.contrib.layers import flatten

def model(x):
    inflated = [20, 48, 56]

    # hyperparameters
    mu = 0
    sigma = 0.1

    #======================================================================
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x20.
    w1 = tf.Variable(tf.truncated_normal([5,5,1,inflated[0]], mean = mu, stddev = sigma))
    b1 = tf.Variable(tf.zeros(inflated[0]))

    l1_conv = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding= 'VALID', name = 'l1_conv')
    l1_conv = tf.nn.bias_add(l1_conv, b1, name = 'l1_conv_bias')

    # activation
    l1_act = tf.nn.relu(l1_conv, name = 'l1_act')

    # Pooling. Input = 28x28x20. Output = 14x14x20.
    l1_pool = tf.nn.max_pool(l1_act, [1,2,2,1],[1,2,2,1],'VALID', name = 'l1_pool')
    l1_pool =tf.nn.dropout(l1_pool,.5)


    #======================================================================
    # Layer 2: Convolutional. Input = 14x14x20. Output = 10x10x48.

    w2 = tf.Variable(tf.truncated_normal([5,5,inflated[0],inflated[1]], mean = mu, stddev = sigma))
    b2 = tf.Variable(tf.zeros(inflated[1]))

    l2_conv = tf.nn.conv2d(l1_pool, w2, strides = [1,1,1,1], padding = 'VALID', name = 'l2_conv')
    l2_conv = tf.nn.bias_add(l2_conv, b2)

    # activation
    l2_act = tf.nn.relu(l2_conv, name = 'l2_act')


    #======================================================================
    # Layer 3: Convolutional. Input = 10x10x48. Output = 6x6x56.

    w3 = tf.Variable(tf.truncated_normal([5,5,inflated[1],inflated[2]], mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros(inflated[2]))

    l3_conv = tf.nn.conv2d(l2_act, w3, strides = [1,1,1,1], padding = 'VALID', name = 'l3_conv')
    l3_conv = tf.nn.bias_add(l3_conv, b3)

    # activation
    l3_act = tf.nn.relu(l3_conv, name = 'l3_act')


    # Pooling. Input = 6x6x56. Output = 3x3x56.
    l3_pool = tf.nn.max_pool(l3_act, [1,2,2,1],[1,2,2,1],'VALID', name = 'l3_pool')
    l3_pool =tf.nn.dropout(l3_pool,.5)


    #===============================================================
    # Flatten. Input = 3x3x56. Output = 400.
    nflat = int(3*3*inflated[2])
    x_flatten = tf.reshape(l3_pool, [-1,nflat], name = 'x_flatten')

    # Layer 4: Fully Connected. Input = 400. Output = 120.
    w4 = tf.Variable(tf.truncated_normal([nflat,120], mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros(120))

    # activation
    l4 = tf.add(tf.matmul(x_flatten,w4), b4, name = 'l4')
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.dropout(l4,.5)

    # Layer 5: Fully Connected. Input = 120. Output = 84.
    w5 = tf.Variable(tf.truncated_normal([120,84], mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(84))

    # activation
    l5 = tf.add(tf.matmul(l4, w5),b5, name = 'l5')
    l5 = tf.nn.relu(l5)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    w6 = tf.Variable(tf.truncated_normal([84,43], mean = mu, stddev = sigma))
    b6 = tf.Variable(tf.zeros(43))

    logits = tf.add(tf.matmul(l5, w6), b6, name = 'l6')
    
    return logits


EPOCHS = 20
BATCH_SIZE = 40
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


from time import time
start_time = time()

# Training
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    num_examples = len(X_train_ini)
#    
#
#    print("Training...")
#    print()
#    for i in range(EPOCHS):
#        X_tref, y_tref = shuffle(X_train_ini, y_train_ini)
#        for offset in range(0, num_examples, BATCH_SIZE):
#            end = offset + BATCH_SIZE
#            batch_x, batch_y = X_tref[offset:end], y_tref[offset:end]
#            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
#
#        validation_accuracy = evaluate(X_valid_ini, y_valid_ini)
#        print("EPOCH {} ...".format(i+1))
#        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#        print()
#
#    print("Final Validation Accuracy = {:.3f}".format(validation_accuracy))
#    saver.save(sess, 'lenet')
#    print("Model saved")



end_time = time()
time_taken = end_time - start_time

hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)

print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")

# Evaluation of test set
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_shuffled, y_test_shuffled)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
















