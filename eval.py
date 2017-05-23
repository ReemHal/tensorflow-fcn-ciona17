import tensorflow as tf
import numpy as np
import os
import os.path
from datetime import datetime
import time
import random
import argparse

import matplotlib.pyplot as plt

from models.vgg6_fc6_512_deconv import vgg16

from utils import input_pipeline_xent, input_pipeline_miou, init_3subplot, update_plots

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_epochs', 15000,
                            """Number of epochs.""")

tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Batch size.""")

tf.app.flags.DEFINE_string('train_record',
                           '/scratch/gallowaa/tfrecord/ciona-16-rgb-702.tfrecords',
                           """Training tfrecord file.""")

tf.app.flags.DEFINE_string('val_record',
                           '/scratch/gallowaa/tfrecord/ciona-16-rgb-334.tfrecords',
                           """Validation tfrecord file.""")

tf.app.flags.DEFINE_string('image_path',
                           '/export/mlrg/gallowaa/Documents/ciona-net-images/multi/vgg7xs-fc6-512-multi',
                           """Image name path""")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--restore', help='where to load model checkpoints from')
    parser.add_argument(
        '--gpu', help='the physical ids of GPUs to use')
    parser.add_argument(
        '--out', help='number of semantic classes', type=int, default=1)
    parser.add_argument(
        '--plot', help='should show predictions every 100 steps')
    parser.add_argument(
        '--model', help='the model variant', default='')
    parser.add_argument(
        '--savepath', default='/home/angus/Documents/CVPR/img/')
    parser.add_argument(
        '--val_record', default='/scratch/gallowaa/tfrecord/ciona-16-rgb-334.tfrecords')

    args = parser.parse_args()

    #outpath = os.path.join(args.train_dir, args.save)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    out_path = 'vgg7' + args.model + '-' + str(args.out) + '-' 

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
            args.val_record, FLAGS.batch_size, shuffle=False)

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

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()  # v0.12

        init = tf.global_variables_initializer()  # v0.12
        init_locals = tf.local_variables_initializer()  # v0.12

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options={'allow_growth': True}))

        sess.run([init, init_locals])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        '''
        summary_writer = tf.summary.FileWriter(outpath, sess.graph)  # v0.12
        training_summary = tf.summary.scalar("train loss", total_loss)  # v0.12
        validation_summary = tf.summary.scalar("val mIoU", mIoU_ph)  # v0.12
        '''

        if args.restore:
            print('Restoring the Network')
            saver.restore(sess, tf.train.latest_checkpoint(args.restore))

        # saver.restore(sess, args.train_dir + '/run3/model.ckpt-99000')

        print('Running the Network')

        if args.plot:
            img_1, img_2, img_3 = init_3subplot()

        num_batches = 334
        miou_list = np.zeros(num_batches / FLAGS.batch_size)

        try:

            step = sess.run(global_step)
            print(step)

            while not coord.should_stop():

                for k in xrange(num_batches):

                    predimg = sess.run(prediction, feed_dict={keep_prob: 1.0})
                    res = sess.run([mIOU], feed_dict={keep_prob: 1.0})
                    print res
                    miou_list[k] = res[0][0]

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

                        fname = out_path + str(k) + '.png'
                        #fname = args.savepath + '-bs' + str(FLAGS.batch_size) + 'stp' + str(step) + '.png'
                        #plt.imsave(fname,img2)
                        plt.savefig(fname)
                        plt.pause(0.05)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
