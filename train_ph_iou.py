import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os
import os.path
from datetime import datetime
import time
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from models.vgg7_fc6_512_deconv import vgg16
#from models.vgg6_fc6_512_deconv import vgg16

from utils import input_pipeline_xent
from utils import input_pipeline_miou
from utils import init_3subplot
from utils import update_plots
from utils import email_results
from utils import get_minibatch

from preprocessing.cielab_float_tfrecords import read_filelist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('num_epochs', 15000,
                            """Number of epochs.""")


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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
        '--train_dir', help='root path for logging events and checkpointing', default='~/logs/vgg7/rgb/iou/')
    parser.add_argument(
        '--restore', help='where to load model checkpoints from')
    parser.add_argument(
        '--gpu', help='the physical ids of GPUs to use')
    parser.add_argument(
        '--out', help='number of semantic classes', type=int, default=1)
    parser.add_argument(
        '--bs', help='batch size', type=int, default=10)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument(
        '--model', help="model variant (either '', 's' - slim filters, 'xs' - extra slim filters)", default='')
    parser.add_argument(
        '--loss', help='loss type (e.g iou loss or cross-entropy)', default='iou')
    parser.add_argument(
        '--fmt', help='input image format (either rgb or lab)', default='lab')
    parser.add_argument(
        '--plot', help='periodically plot validation progress during training')
    parser.add_argument(
        '--root_path', help="path to images", default="/export/mlrg/gallowaa/Documents/ciona-net/data/")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    parser.add_argument(
        '--email', help="should send an email with results when job finished", action="store_true")

    args = parser.parse_args()

    outpath = os.path.join(args.train_dir, args.save)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get list of training and validation images
    train_path = os.path.join(args.root_path, 'data-train-ciona-16/')
    val_path = os.path.join(args.root_path, 'data-val-ciona-16/')

    t_lumList, t_alphaList, t_betaList, t_segList = read_filelist(train_path)
    v_lumList, v_alphaList, v_betaList, v_segList = read_filelist(val_path)

    print('args.out = %d' % args.out)

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32)
        mIoU_ph = tf.placeholder(tf.float32)

        p_cielab = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='p_cielab')
        p_mask = tf.placeholder(tf.float32, shape=[None, 224, 224], name='p_mask')

        print('p_mask shape')
        print(p_mask.get_shape())

        labels = tf.expand_dims(p_mask, 3, name='labels')

        print('labels shape')
        print(labels.get_shape())

        filter_dims_arr = np.array([[16, 32, 64, 128, 256, 512],    # vgg#xs
                                    [32, 64, 128, 256, 512, 512],   # vgg#s
                                    [64, 128, 256, 512, 512, 512]])  # vgg

        # This would normally be done in `read_and_decode_xent` part of pipeline
        cielab = p_cielab * 1.0 / 255.0

        '''
        If only two classes, merge classes 0 (other) and 1 (mussel), otherwise do nothing.
        If args.loss = 'iou', we also want to binarize labels, but args.out is actually 1 so that
        model only uses one output layer (sigmoid)
        '''
        #if args.out == 2 or args.loss == 'iou':

        # greater_equal (bool)
        labels = tf.cast((tf.greater_equal(labels, 2)), tf.int32, name='labels_int32')

        if args.model == 'xs':
            vgg = vgg16(args.out, filter_dims_arr[0, :], cielab, keep_prob)
        elif args.model == 's':
            vgg = vgg16(args.out, filter_dims_arr[1, :], cielab, keep_prob)
        else:
            vgg = vgg16(args.out, filter_dims_arr[2, :], cielab, keep_prob)

        logits = vgg.up
        labels = tf.reshape(labels, [-1])

        # Approximate IoU loss from
        # http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
        logits = tf.reshape(logits, [-1])

        # binarize the network output
        prediction = tf.cast(tf.greater_equal(logits, 0.5), tf.int32)

        #trn_labels = tf.reshape(trn_labels_batch, [-1])
        labels_f32 = tf.cast(labels, tf.float32)
        inter = tf.reduce_sum(tf.multiply(logits, labels_f32, name='mul_b_sum'))
        union = tf.reduce_sum(
            tf.subtract(tf.add(logits, labels_f32), tf.multiply(logits, labels_f32)))

        total_loss = tf.subtract(tf.constant(
            1.0, dtype=tf.float32), tf.div(inter, union))

        mIOU = tf.contrib.metrics.streaming_mean_iou(prediction, labels, 2)

        train_op = tf.train.AdamOptimizer(args.lr).minimize(
            total_loss, global_step=global_step)

        ''' # This doesn't make sense for binary iou - output is sigmoid
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
        
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
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
        max_valid = 0
        print('About to try, patience = %d' % patience)

        try:
            step = sess.run(global_step)
            print(step)
            while not coord.should_stop() and step < args.max_steps and patience < args.max_patience:

                start_time = time.time()

                mb = get_minibatch(args.bs, t_lumList,
                                    t_alphaList, t_betaList, t_segList)

                my_mask = mb[:, :, :, 3]

                _, train_loss, train_summ = sess.run(
                    [train_op, total_loss, training_summary],
                    feed_dict={keep_prob: 0.5, p_cielab: mb[:, :, :, 0:3], p_mask: my_mask})

                duration = time.time() - start_time

                assert not np.isnan(
                    train_loss), 'Model diverged with loss = NaN'

                # Training
                if step % 10 == 0:

                    #num_examples_per_step = args.bs
                    examples_per_sec = args.bs / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, train=%.5f (%.1f ex/sec)')
                    print(format_str % (datetime.now(), step, train_loss, examples_per_sec))
                    
                    summary_writer.add_summary(train_summ, step)
                    #summary_str = sess.run(summary_op)
                    #summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    
                '''
                if args.plot:
                    if step % 100 == 0:
                        myimg, mylbl = sess.run(
                            [val_images_batch, val_labels_batch])
                        predimg = sess.run(valid_prediction, feed_dict={
                                           keep_prob: 1.0, p_lumi: lumi, p_alph: alph, p_beta: beta, p_mask: mask})

                        # print '%.4f,' % miou_list[1:].mean()

                        update_plots(myimg, predimg, mylbl, args.bs)

                        fname = '/export/mlrg/gallowaa/Documents/ciona-net-images/multi/vgg7xs-fc6-512-multi' + \
                            '-bs' + str(args.bs) + 'stp' + str(step) + '.png'
                        # plt.imsave(fname,img2)
                        # plt.savefig(fname)

                        plt.pause(0.3)
                '''
                # Run mIoU on entire validation set every 500 steps
                if step % 500 == 0 and step > 0:

                    for k in range(int(num_batches / args.bs)):

                        mb = get_minibatch(
                            args.bs, v_lumList, v_alphaList, v_betaList, v_segList)
                        res = sess.run(mIOU, feed_dict={keep_prob: 1.0, p_cielab: mb[
                                       :, :, :, 0:3], p_mask: mb[:, :, :, 3]})

                        print("i=%d, IoU=%.4f" % (k, res[0]))
                        miou_list[k] = res[0]
                    curr_miou = miou_list[1:].mean()
                    print('Adding mIoU summary... %.4f' % curr_miou)

                    validation_summ = sess.run(validation_summary, feed_dict={mIoU_ph: curr_miou})
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

                    if curr_miou > max_valid:
                        max_valid = curr_miou

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