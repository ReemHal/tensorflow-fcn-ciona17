import tensorflow as tf
import numpy as np
import os
import os.path
from datetime import datetime
import time
import random
import argparse

import matplotlib.pyplot as plt

from models.vgg7_fc6_512_deconv import vgg16

from utils import input_pipeline_xent, input_pipeline_miou, init_3subplot, update_plots

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path', help='path to checkpoints folder')
    parser.add_argument(
        '--restore', help='specific checkpoint to use (e.g model.ckpt-99000), otherwise use latest')
    parser.add_argument(
        '--gpu', help='the physical ids of GPUs to use')
    parser.add_argument(
        '--out', help='number of semantic classes', type=int, default=1)
    parser.add_argument(
        '--fmt', help='input image format (either rgb or lab)', default='rgb')
    parser.add_argument(
        '--model', help='the model variant (should match checkpoint otherwise will crash)', default='')
    parser.add_argument(
        '--savepath', default='~/Documents/img/')
    parser.add_argument(
        '--plot', help='periodically plot validation progress during training', action="store_true")
    parser.add_argument(
        '--val_record', default='~/tfrecord/ciona-17-rgb-valid.tfrecords')

    args = parser.parse_args()

    #outpath = os.path.join(args.train_dir, args.save)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #out_path = 'vgg7' + args.model + '-' + str(args.out) + '-' 

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32)
        mIoU_ph = tf.placeholder(tf.float32)

        # Cross-entropy loss
        '''
        if args.loss == 'xent':
            val_images_batch, val_labels_batch = input_pipeline_xent(
                args.val_record, FLAGS.batch_size, args.out, shuffle=False)
        else:
        '''
        val_images_batch, val_labels_batch = input_pipeline_miou(
            args.val_record, FLAGS.batch_size, args.fmt, shuffle=False)

        images_ph = tf.placeholder_with_default(
            val_images_batch, shape=[None, 224, 224, 3])
        labels_ph = tf.placeholder_with_default(
            val_labels_batch, shape=[None, 224, 224, 1])

        filter_dims_arr = np.array([[16, 32, 64, 128, 256, 512],    # vgg#xs
                                    [32, 64, 128, 256, 512, 512],   # vgg#s
                                    [64, 128, 256, 512, 512, 512]])  # vgg

        if args.model == 'xs':
            vgg = vgg16(args.out, filter_dims_arr[0, :], images_ph, keep_prob)
        elif args.model == 's':
            vgg = vgg16(args.out, filter_dims_arr[1, :], images_ph, keep_prob)
        else:
            vgg = vgg16(args.out, filter_dims_arr[2, :], images_ph, keep_prob)

        # vgg = vgg16(args.out, images_ph, keep_prob)

        logits = vgg.up
        labels = tf.reshape(labels_ph, [-1])
        logits = tf.reshape(logits, [-1])

        # binarize the network output

        #prediction = tf.greater_equal(logits, 0.5)
        prediction = tf.cast(tf.greater_equal(
            logits, 0.5, name='thresh'), tf.int32)
        #trn_labels = tf.reshape(trn_labels_batch, [-1])

        inter = tf.reduce_sum(tf.multiply(logits, labels))
        union = tf.reduce_sum(
            tf.subtract(tf.add(logits, labels), tf.multiply(logits, labels)))

        total_loss = tf.subtract(tf.constant(
            1.0, dtype=tf.float32), tf.div(inter, union))

        mIOU = tf.contrib.metrics.streaming_mean_iou(prediction, labels, 2)

        '''
        valid_prediction = tf.argmax(tf.reshape(
            tf.nn.softmax(logits), tf.shape(vgg.up)), dimension=3)
        '''

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())  # v0.12

        init = tf.global_variables_initializer()  # v0.12
        init_locals = tf.local_variables_initializer()  # v0.12

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options={'allow_growth': True}))

        sess.run([init, init_locals])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        if args.restore:
            print('Restoring the network from %s' % os.path.join(args.path, args.restore))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.path, args.restore)))
        else:    
            print('Restoring the network from path %s' % args.path)
            saver.restore(sess, tf.train.latest_checkpoint(args.path))

        # saver.restore(sess, args.train_dir + '/run3/model.ckpt-99000')

        print('Running the Network')

        if args.plot:
            img_1, img_2, img_3 = init_3subplot()

        num_batches = 334
        miou_list = np.zeros(int(num_batches / FLAGS.batch_size))

        try:

            step = sess.run(global_step)
            print(step)

            while not coord.should_stop():

                for k in range(num_batches):

                    predimg = sess.run(prediction, feed_dict={keep_prob: 1.0})
                    '''
                    res = sess.run([mIOU], feed_dict={keep_prob: 1.0})
                    print(res)
                    miou_list[k] = res[0][0]
                    '''
                    if args.plot:

                        myimg, mylbl = sess.run(
                            [val_images_batch, val_labels_batch])
                        '''
                        predimg = sess.run(valid_prediction, feed_dict={
                        images_ph: myimg, labels_ph: mylbl, keep_prob: 1.0})
                        '''

                        # print '%.4f,' % miou_list[1:].mean()

                        update_plots(img_1, img_2, img_3, myimg, predimg, mylbl,
                                     FLAGS.batch_size, args.out)

                        #fname = out_path + str(k) + '.png'
                        #fname = args.savepath + '-bs' + str(FLAGS.batch_size) + 'stp' + str(step) + '.png'
                        #plt.imsave(fname,img2)
                        #plt.savefig(fname)
                        plt.pause(0.05)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
