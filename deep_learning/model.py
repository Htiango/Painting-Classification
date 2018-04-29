import tensorflow as tf
import numpy as np
import time
from dataset import DataSet

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 100

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# try 2 styles
classes = ['realism', 'abstract-art']
num_classes = len(classes)

# batch size
batch_size = 32

# validation split
validation_size = .16

checkpointsPath = "./models"

def get_arr(y):
    y_arr = np.zeros((y.shape[0], num_classes))
    for i in range(y.shape[0]):
        val = y[i]
        arr = np.zeros((num_classes))
        arr[val] = 1
        y_arr[i,:] = arr
    return y_arr


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              
                   num_input_channels, 
                   filter_size,        
                   num_filters,       
                   use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    # keep_prob = 0.5
    # layer = tf.nn.dropout(layer, keep_prob)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          
                 num_inputs,     
                 num_outputs,    
                 use_relu=True): 
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def build_model():
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)
    layer_flat, num_features = flatten_layer(layer_conv3)
    layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
    layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                        labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y_true, cost, optimizer, accuracy, y_pred_cls


def train(X_tr, y_tr, num_iterations, reload=True):
    x, y_true, cost, optimizer, accuracy, y_pred_cls= build_model()
    
    tr_num = X_tr.shape[0]
    va_idx_end = int(validation_size*tr_num)
    X_va = X_tr[:va_idx_end]
    y_va = y_tr[:va_idx_end]
    X_tr = X_tr[va_idx_end:]
    y_tr = y_tr[va_idx_end:]
    data_train = DataSet(X_tr, get_arr(y_tr), y_tr, classes)
    data_valid = DataSet(X_va, get_arr(y_va), y_va, classes)

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_batch_size = batch_size

        def print_progress(epoch, acc, val_acc, val_loss):
            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss))

        acc_tr_sum = 0
        acc_va_sum = 0
        loss_va_sum = 0
        num_sum = 0

        saver = tf.train.Saver()

        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")

        for i in range(num_iterations):
            x_batch, y_true_batch, cls_batch = data_train.next_batch(train_batch_size)
            x_valid_batch, y_valid_batch, valid_cls_batch = data_valid.next_batch(train_batch_size)

            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            
            feed_dict_validate = {x: x_valid_batch,
                                  y_true: y_valid_batch}

            sess.run(optimizer, feed_dict=feed_dict_train)
            
            acc_tr_sum += sess.run(accuracy, feed_dict=feed_dict_train)
            acc_va_sum += sess.run(accuracy, feed_dict=feed_dict_validate)
            loss_va_sum += sess.run(cost, feed_dict=feed_dict_validate)
            num_sum += 1

            if i % int(data_train.num_examples/batch_size) == 0: 
                epoch = int(i / int(data_train.num_examples/batch_size))            
                print_progress(epoch, acc_tr_sum / num_sum, acc_va_sum / num_sum, loss_va_sum / num_sum)
                acc_tr_sum = 0
                acc_va_sum = 0
                loss_va_sum = 0
                num_sum = 0
                if epoch % 5 == 0:
                    save_path = saver.save(sess,checkpointsPath + "/model",global_step=epoch)
                    print("Model saved in path: %s" % save_path)

    end_time = time.time()
    time_dif = end_time - start_time

    print("Finish Training! \nTime elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

def test(X_te, y_te):
    data_test = DataSet(X_te, get_arr(y_te), y_te, classes)

    x, y_true, cost, optimizer, accuracy, y_pred_cls = build_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath + "/model-realism-abstract")
        # if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            print(checkPoint.model_checkpoint_path)
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")
            exit(0)

        num_test = len(data_test.images)
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        i = 0

        while i < num_test:
            j = min(i + batch_size, num_test)

            images = data_test.images[i:j, :]
            labels = data_test.labels[i:j, :]
            feed_dict = {x: images,
                         y_true: labels}

            cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
            i = j

        cls_true = np.array(data_test.label_texts)
        cls_pred = np.array([classes[x] for x in cls_pred]) 

        correct = (cls_true == cls_pred)
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test

        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

