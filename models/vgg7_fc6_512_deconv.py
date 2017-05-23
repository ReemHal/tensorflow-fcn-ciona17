
import tensorflow as tf
import numpy as np


class vgg16:

    def __init__(self, n_classes, f, imgs=None, keep_prob=None, weights=None, skip_layers=None):
        self.imgs = imgs
        self.keep_prob = keep_prob
        self.convlayers(n_classes, f)
        # self.fc_layers()
        #self.probs = self.logits
        self.skip_layers = skip_layers

        if weights is not None and skip_layers is not None:
            #self.load_weights(weights, sess)
            self.load_with_skip(weights, self.skip_layers)

    def convlayers(self, n_classes, f):
        self.parameters = []

        # zero-mean input
        # with tf.name_scope('preprocess') as scope:
        #    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #    images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, f[0]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[0]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, f[0], f[1]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[1]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, f[1], f[2]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[2]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, f[2], f[3]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[3]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, f[3], f[4]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[4]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # output of self.conv5_3 = [batch_size, 14,14, 512]
        self.pool5 = tf.nn.max_pool(self.conv5_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')

        with tf.name_scope('fc6') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, f[4], f[5]], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool5, kernel, [
                                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[f[5]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.fc6 = tf.nn.dropout(tf.nn.relu(out, name=scope),
                                     self.keep_prob)
            self.parameters += [kernel, biases]

        with tf.name_scope('upscore') as scope:
            kernel = tf.Variable(tf.truncated_normal([64, 64, n_classes, f[5]], dtype=tf.float32,  # 2 here is N output classes
                                                     stddev=1e-1), name='weights')

            shape = tf.shape(self.imgs)
            # 2 here is N output classes
            new_shape = [shape[0], shape[1], shape[2], n_classes]
            output_shape = tf.stack(new_shape)
            deconv = tf.nn.conv2d_transpose(self.fc6, kernel, output_shape, [
                                            1, 32, 32, 1], padding='SAME')
            if n_classes == 1:
                self.up = tf.sigmoid(deconv, name=scope)
            else:
                self.up = tf.nn.relu(deconv, name=scope)
                
            self.parameters += [kernel]
            self.pred_up = tf.argmax(self.up, dimension=3)

    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            self.parameters[i].assign(weights[k])

    def load_with_skip(self, weight_file, skip_layer):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if k not in skip_layer:
                print(i, k, np.shape(weights[k]))
                self.parameters[i].assign(weights[k])
