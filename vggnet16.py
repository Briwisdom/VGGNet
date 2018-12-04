# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as sciio
import numpy as np

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def conv(name, x, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def fc(name, x, w, b):
    return tf.nn.relu(tf.matmul(x, w) + b, name=name)


def vgg_net(_X, _weights, _biases, keep_prob):
    conv1_1 = conv('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'])
    conv1_2 = conv('conv1_2', conv1_1, _weights['wc1_2'], _biases['bc1_2'])
    pool1 = max_pool('pool1', conv1_2, k=2)

    conv2_1 = conv('conv2_1', pool1, _weights['wc2_1'], _biases['bc2_1'])
    conv2_2 = conv('conv2_2', conv2_1, _weights['wc2_2'], _biases['bc2_2'])
    pool2 = max_pool('pool2', conv2_2, k=2)

    conv3_1 = conv('conv3_1', pool2, _weights['wc3_1'], _biases['bc3_1'])
    conv3_2 = conv('conv3_2', conv3_1, _weights['wc3_2'], _biases['bc3_2'])
    conv3_3 = conv('conv3_3', conv3_2, _weights['wc3_3'], _biases['bc3_3'])
    pool3 = max_pool('pool3', conv3_3, k=2)

    conv4_1 = conv('conv4_1', pool3, _weights['wc4_1'], _biases['bc4_1'])
    conv4_2 = conv('conv4_2', conv4_1, _weights['wc4_2'], _biases['bc4_2'])
    conv4_3 = conv('conv4_3', conv4_2, _weights['wc4_3'], _biases['bc4_3'])
    pool4 = max_pool('pool4', conv4_3, k=2)

    # conv5_1 = conv('conv5_1', pool4, _weights['wc5_1'], _biases['bc5_1'])
    # conv5_2 = conv('conv5_2', conv5_1, _weights['wc5_2'], _biases['bc5_2'])
    # conv5_3 = conv('conv5_3', conv5_2, _weights['wc5_3'], _biases['bc5_3'])
    # pool5 = max_pool('pool5', conv5_3, k=2)

    _shape = pool4.get_shape()
    flatten = _shape[1].value * _shape[2].value * _shape[3].value
    pool5 = tf.reshape(pool4, shape=[-1, flatten])
    fc1 = fc('fc1', pool5, _weights['fc1'], _biases['fb1'])
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = fc('fc2', fc1, _weights['fc2'], _biases['fb2'])
    fc2 = tf.nn.dropout(fc2, keep_prob)

    fc3 = fc('fc3', fc2, _weights['fc3'], _biases['fb3'])
    fc3 = tf.nn.dropout(fc3, keep_prob)

    out = tf.nn.softmax(fc3)

    return out

weights={
    'wc1_1':tf.Variable(tf.random_normal([3,3,1,16])),
    'wc1_2':tf.Variable(tf.random_normal([3,3,16,16])),
    'wc2_1':tf.Variable(tf.random_normal([3,3,16,32])),
    'wc2_2':tf.Variable(tf.random_normal([3,3,32,32])),
    'wc3_1':tf.Variable(tf.random_normal([3,3,32,64])),
    'wc3_2':tf.Variable(tf.random_normal([3,3,64,64])),
    'wc3_3':tf.Variable(tf.random_normal([3,3,64,64])),
    'wc4_1':tf.Variable(tf.random_normal([3,3,64,128])),
    'wc4_2':tf.Variable(tf.random_normal([3,3,128,128])),
    'wc4_3':tf.Variable(tf.random_normal([3,3,128,128])),
    'wc5_1':tf.Variable(tf.random_normal([3,3,128,128])),
    'wc5_2':tf.Variable(tf.random_normal([3,3,128,128])),
    'wc5_3':tf.Variable(tf.random_normal([3,3,128,128])),
    'fc1':tf.Variable(tf.random_normal([3*3*128,128])),
    'fc2':tf.Variable(tf.random_normal([128,128])),
    'fc3':tf.Variable(tf.random_normal([128,7]))
}

biases={
    'bc1_1':tf.Variable(tf.random_normal([16])),
    'bc1_2':tf.Variable(tf.random_normal([16])),
    'bc2_1':tf.Variable(tf.random_normal([32])),
    'bc2_2':tf.Variable(tf.random_normal([32])),
    'bc3_1':tf.Variable(tf.random_normal([64])),
    'bc3_2':tf.Variable(tf.random_normal([64])),
    'bc3_3':tf.Variable(tf.random_normal([64])),
    'bc4_1':tf.Variable(tf.random_normal([128])),
    'bc4_2':tf.Variable(tf.random_normal([128])),
    'bc4_3':tf.Variable(tf.random_normal([128])),
    'bc5_1':tf.Variable(tf.random_normal([128])),
    'bc5_2':tf.Variable(tf.random_normal([128])),
    'bc5_3':tf.Variable(tf.random_normal([128])),
    'fb1': tf.Variable(tf.random_normal([128])),
    'fb2': tf.Variable(tf.random_normal([128])),
    'fb3': tf.Variable(tf.random_normal([7]))
}

def train(learning_rate = 0.001,batch_size = 500,max_iters = 20000,sample_size=48):
    x = tf.placeholder(tf.float32, [None, sample_size,sample_size,1])
    y = tf.placeholder(tf.float32, [None, 7])
    keep_prob = tf.placeholder(tf.float32)

    pred = vgg_net(x, weights, biases, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    target = sciio.loadmat('E:/ice_experiment_lmh/Sequential/icematdata/resnet_48pixel-7class')
    train_x = target['train_x']
    test_x = target['test_x']
    train_y = target['train_y']
    test_y = target['test_y']
    print('read data finished')
    print('start training...')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        test_x=np.reshape(test_x,[np.shape(test_x)[0],sample_size,sample_size,1])
        while step * batch_size < max_iters:
            step = 1
            j = 0
            for batch_xs, batch_ys in minibatches(train_x, train_y, batch_size, shuffle=True):
                batch_xs = np.reshape(batch_xs, [batch_size, sample_size, sample_size, 1])
                _, acc = sess.run([optimizer, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
                print('the step %s acc is: %s ' % (j, "{:.6f}".format(acc)))
                j += 1
            test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
            print("-----------------")
            print('the epoch %s acc is: %s ' % (step, "{:.6f}".format(test_acc)))
            print('-------------------')
            step += 1
        print('train over !')

if __name__ == '__main__':
    train()
