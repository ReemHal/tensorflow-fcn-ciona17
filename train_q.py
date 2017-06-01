import tensorflow as tf
import numpy as np
import os
import os.path
from datetime import datetime
import time
import random
import argparse

import matplotlib.pyplot as plt

#from models.vgg7s_fc6_512_deconv_multi import vgg16
#from models.vgg7xs_fc6_512_deconv_multi import vgg16
#from models.vgg12xs_fc6_512_deconv_multi import vgg16
#from models.vgg12_fc6_512_deconv_multi import vgg16
#from models.vgg7_fc6_512_deconv_multi import vgg16

from models.vgg7_fc6_512_deconv import vgg16
#from models.vgg6_fc6_512_deconv import vgg16

from utils import input_pipeline_xent
from utils import input_pipeline_miou
from utils import init_3subplot
from utils import update_plots
from utils import email_results

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('num_epochs', 15000,
                            """Number of epochs.""")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'sub', help='Sub-directory under --train_dir for logging events and checkpointing.   \
        Would usually give a unique name (e.g initial learning rate used) so that tensorboard \
        results are more easily interpreted')
    parser.add_argument(
        '--max_steps', help='maximum number of steps for training', type=int, default=50000)
    parser.add_argument(
        '--max_patience', help='number of consecutive times validation error is allowed to decrease before stopping', type=int, default=10)
    parser.add_argument(
        '--train_dir', help='root path for logging events and checkpointing', default='logs/vgg7/rgb/iou/')
    parser.add_argument(
        '--restore', help='path to load model checkpoints from')
    parser.add_argument(
        '--gpu', help='physical id of GPUs to use')
    parser.add_argument(
        '--out', help='number of semantic classes', type=int, default=1)
    parser.add_argument(
        '--bs', help='batch size', type=int, default=1)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument(
        '--model', help="model variant (either '', 's' - slim filters, 'xs' - extra slim filters)", default='')
    parser.add_argument(
        '--loss', help='loss type (e.g iou loss or cross-entropy)', default='iou')
    parser.add_argument(
        '--fmt', help='input image format (either rgb or lab)', default='rgb')
    parser.add_argument(
        '--plot', help='periodically plot validation progress during training', action="store_true")
    parser.add_argument(
        '--train_record', default='tfrecord/ciona-17-rgb-train.tfrecords')
    parser.add_argument(
        '--val_record', default='tfrecord/ciona-17-rgb-valid.tfrecords')
    parser.add_argument(
        '--email', help="send an email with results when job finished, requires .env", action="store_true")

    args = parser.parse_args()

    outpath = os.path.join(args.train_dir, args.sub)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32)
        mIoU_ph = tf.placeholder(tf.float32)

        # Cross-entropy loss
        if args.loss == 'xent':
            trn_images_batch, trn_labels_batch = input_pipeline_xent(
                args.train_record, args.bs, args.out, shuffle=True)

            val_images_batch, val_labels_batch = input_pipeline_xent(
                args.val_record, args.bs, args.out, shuffle=False)
        else:
            trn_images_batch, trn_labels_batch = input_pipeline_miou(
                args.train_record, args.bs, args.fmt, shuffle=True)

            val_images_batch, val_labels_batch = input_pipeline_miou(
                args.val_record, args.bs, args.fmt, shuffle=False)

        images_ph = tf.placeholder_with_default(
            trn_images_batch, shape=[None, 224, 224, 3])
        labels_ph = tf.placeholder_with_default(
            trn_labels_batch, shape=[None, 224, 224, 1])

        filter_dims_arr = np.array([[16, 32, 64, 128, 256, 512],    # vgg#xs
                                    [32, 64, 128, 256, 512, 512],   # vgg#s
                                    [64, 128, 256, 512, 512, 512]])  # vgg

        if args.model == 'xs':
            vgg = vgg16(args.out, filter_dims_arr[0, :], images_ph, keep_prob)
        elif args.model == 's':
            vgg = vgg16(args.out, filter_dims_arr[1, :], images_ph, keep_prob)
        else:
            vgg = vgg16(args.out, filter_dims_arr[2, :], images_ph, keep_prob)

        #vgg = vgg16(args.out, images_ph, keep_prob)

        logits = vgg.up

        labels = tf.reshape(labels_ph, [-1])

        # Cross-entropy loss
        if args.loss == 'xent':

            logits = tf.reshape(logits, (-1, args.out))
            prediction = tf.argmax(tf.nn.softmax(logits), dimension=1)

            output = tf.nn.softmax(logits)

            labels = tf.cast(labels, tf.int64)
            labels = tf.minimum(labels, 2)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)

            total_loss = tf.reduce_mean(cross_entropy, name='x_ent_mean')

            mIOU = tf.contrib.metrics.streaming_mean_iou(
                tf.cast(prediction, tf.int32), labels, args.out)

        # Approximate IoU loss from
        # http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
        else:

            logits = tf.reshape(logits, [-1])
            # binarize the network output
            prediction = tf.greater_equal(logits, 0.5)
            #trn_labels = tf.reshape(trn_labels_batch, [-1])

            inter = tf.reduce_sum(tf.multiply(logits, labels))
            union = tf.reduce_sum(
                tf.subtract(tf.add(logits, labels), tf.multiply(logits, labels)))

            total_loss = tf.subtract(tf.constant(
                1.0, dtype=tf.float32), tf.div(inter, union))

            mIOU = tf.contrib.metrics.streaming_mean_iou(
                tf.cast(prediction, tf.int32), labels, 2)

        # total_loss = loss(logits, trn_labels_batch, args.out,
        # head=None)

        train_op = tf.train.AdamOptimizer(args.lr).minimize(
            total_loss, global_step=global_step)

        valid_prediction = tf.argmax(tf.reshape(
            tf.nn.softmax(logits), tf.shape(vgg.up)), dimension=3)

        '''
        mIOU = tf.contrib.metrics.streaming_mean_iou(
            tf.cast(prediction, tf.int32), labels, args.out)
        '''

        # accuracy=tf.reduce_sum(tf.pow(valid_prediction - labels, 2))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())  # v0.12

        # Build the summary operation based on the TF collection of Summaries.
        #summary_op = tf.summary.merge_all()  # v0.12

        init = tf.global_variables_initializer()  # v0.12
        init_locals = tf.local_variables_initializer()  # v0.12

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options={'allow_growth': True}))

        sess.run([init, init_locals])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        summary_writer = tf.summary.FileWriter(outpath, sess.graph)  # v0.12
        training_summary = tf.summary.scalar("train loss", total_loss)  # v0.12
        validation_summary = tf.summary.scalar("val mIoU", mIoU_ph)  # v0.12

        if args.restore:
            print('Restoring the Network')
            saver.restore(sess, tf.train.latest_checkpoint(
                os.path.join(args.train_dir, args.restore)))

        # saver.restore(sess, args.train_dir + '/run3/model.ckpt-99000')

        print('Running the Network')

        if args.plot:
            img_1, img_2, img_3 = init_subplot()

        num_batches = 334
        miou_list = np.zeros(int(num_batches / args.bs))

        patience = 0
        prv_miou = 0
        curr_miou = 0
        print('About to try, patience = %d' % patience)

        try:
            step = sess.run(global_step)
            print(step)
            while not coord.should_stop() and step < args.max_steps and patience < args.max_patience:

                start_time = time.time()
                _, train_loss, train_summ = sess.run(
                    [train_op, total_loss, training_summary],
                    feed_dict={keep_prob: 0.5})
                _, train_loss = sess.run([train_op, total_loss],
                                         feed_dict={keep_prob: 0.5})
                duration = time.time() - start_time

                assert not np.isnan(
                    train_loss), 'Model diverged with loss = NaN'

                # Training
                if step % 10 == 0:

                    num_examples_per_step = args.bs
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, train=%.6f, (%.1f ex/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, train_loss,
                                        examples_per_sec, sec_per_batch))
                    summary_writer.add_summary(train_summ, step)
                    '''
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    '''
                    summary_writer.flush()

                if args.plot:
                    if step % 100 == 0:
                        myimg, mylbl = sess.run(
                            [val_images_batch, val_labels_batch])
                        predimg = sess.run(valid_prediction, feed_dict={
                                           images_ph: myimg, labels_ph: mylbl, keep_prob: 1.0})

                        # print '%.4f,' % miou_list[1:].mean()

                        update_plots(myimg, predimg, mylbl, args.bs)

                        '''
                        fname = '/export/mlrg/gallowaa/Documents/ciona-net-images/multi/vgg7xs-fc6-512-multi' + \
                            '-bs' + str(args.bs) + 'stp' + str(step) + '.png'
                           '''
                        # plt.imsave(fname,img2)
                        # plt.savefig(fname)

                        plt.pause(0.3)

                # Run mIoU on entire validation set every 500 steps
                if step % 500 == 0 and step > 0:

                    for k in range(int(num_batches / args.bs)):

                        myimg, mylbl = sess.run(
                            [val_images_batch, val_labels_batch])
                        res = sess.run(mIOU, feed_dict={
                                       images_ph: myimg, labels_ph: mylbl, keep_prob: 1.0})
                        print("i=%d, IoU=%.4f" % (k, res[0]))
                        miou_list[k] = res[0]
                    curr_miou = miou_list[1:].mean()
                    print('Adding mIoU summary... %.4f' % curr_miou)
                    validation_summ = sess.run(validation_summary, feed_dict={
                                               mIoU_ph: miou_list[1:].mean()})
                    summary_writer.add_summary(validation_summ, step)
                    '''
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    '''
                    summary_writer.flush()

                if curr_miou < prv_miou:
                    patience += 1

                if curr_miou > prv_miou:
                    patience = 0

                prv_miou = curr_miou

                # Save the model checkpoint periodically.

                if step % 5000 == 0 and step > 0:
                    checkpoint_path = os.path.join(outpath, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

    if args.email:
        email_results(step, max_valid)
