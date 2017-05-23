
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

# email related imports
import os
import socket
import smtplib
from email.mime.text import MIMEText


def read_and_decode_lab(filename_queue):
    """Read and decode CIELAB data

    Keyword arguments:
    filename_queue -- name of queue to read from
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            # when line 21 = tf.string => Name: <unknown>, Key: lumi_raw,
            # Index: 0.  Data types don't match. Data type: floatExpected type:
            # string
            'lumi_raw': tf.FixedLenFeature((), tf.float32),
            'alph_raw': tf.FixedLenFeature((), tf.float32),
            'beta_raw': tf.FixedLenFeature((), tf.float32),
            'mask_raw': tf.FixedLenFeature((), tf.string),
        })

    # returns a vector of type tf.float32
    # when line 21 = tf.float32 => TypeError: Input 'bytes' of 'DecodeRaw' Op
    # has type float32 that does not match expected type of string.
    lumi = tf.decode_raw(features['lumi_raw'], tf.float32, little_endian=True)
    lumi.set_shape([224 * 224])
    lumi = tf.reshape(lumi, [224, 224])

    alph = tf.decode_raw(features['alph_raw'], tf.float32)
    alph.set_shape([224 * 224])
    alph = tf.reshape(alph, [224, 224])

    beta = tf.decode_raw(features['beta_raw'], tf.float32)
    beta.set_shape([224 * 224])
    beta = tf.reshape(beta, [224, 224])

    image = tf.stack([lumi, alph, beta], axis=2)
    lab = tf.cast(image, tf.float32)

    label = tf.decode_raw(features['mask_raw'], tf.uint8)
    label.set_shape([224 * 224 * 1])
    label = tf.reshape(label, [224, 224, 1])

    # possibly insert flag here to check for loss type (xent or iou)
    label = tf.cast((tf.greater_equal(label, 2)), tf.int32)
    label = tf.cast(label, tf.float32)

    # label = tf.cast((tf.greater_equal(label, 2)), tf.int64)
    lab = lab * 1.0 / 255.0

    return lab, label


def read_and_decode_miou(filename_queue):
    """Read and decode for multi class iou loss (cast label as int32)

    Keyword arguments:
    filename_queue -- name of queue to read from
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([224 * 224 * 3])
    image = tf.reshape(image, [224, 224, 3])

    label = tf.decode_raw(features['mask_raw'], tf.uint8)
    label.set_shape([224 * 224 * 1])
    label = tf.reshape(label, [224, 224, 1])

    label = tf.cast((tf.greater_equal(label, 2)), tf.int32)
    label = tf.cast(label, tf.float32)

    bgr = tf.cast(image, tf.float32)
    bgr = bgr * 1.0 / 255.0

    return bgr, label


def read_and_decode_xent(filename_queue):
    """Read and decode for binary cross-entropy loss (cast label as int64)

    Keyword arguments:
    filename_queue -- name of queue to read from
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([224 * 224 * 3])
    image = tf.reshape(image, [224, 224, 3])

    label = tf.decode_raw(features['mask_raw'], tf.uint8)
    label.set_shape([224 * 224 * 1])
    label = tf.reshape(label, [224, 224, 1])

    bgr = tf.cast(image, tf.float32)

    label = tf.cast((tf.greater_equal(label, 2)), tf.int64)
    bgr = bgr * 1.0 / 255.0

    return bgr, label


def read_and_decode_xent_multi(filename_queue, noise):
    """Read and decode when using multi output cross-entopy loss

    Keyword arguments:
    filename_queue -- name of queue to read from
    noise -- Should noise be added?
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([224 * 224 * 3])
    image = tf.reshape(image, [224, 224, 3])

    label = tf.decode_raw(features['mask_raw'], tf.uint8)
    label.set_shape([224 * 224 * 1])
    label = tf.reshape(label, [224, 224, 1])

    if noise is True:
        image = tf.image.random_brightness(image, 0.15, seed=None)
        image = tf.image.random_contrast(image, 0.15, 0.25, seed=None)

    bgr = tf.cast(image, tf.float32)

    # label = tf.cast(label, tf.int64)
    # label = tf.select(label != 255, label, -1 * tf.ones_like(label))
    # label = tf.reshape(label, [224, 224, 1])
    # label = tf.minimum(label, 2)
    bgr = bgr * 1.0 / 255.0

    return bgr, label


def input_pipeline_xent(filenames, batch_size, n_classes, num_epochs=None, shuffle=True):
    """The input pipeline when using cross-entropy loss

    Keyword arguments:
    filenames -- name of queue to read from
    batch_size -- the mini-batch size
    n_classes -- number of output classes
    num_epochs -- max number of epochs
    shuffle -- should data be shuffled
    """

    filename_queue = tf.train.string_input_producer(
        [filenames], num_epochs=num_epochs, shuffle=shuffle)

    if n_classes > 2:
        image, label = read_and_decode_xent_multi(filename_queue, noise=False)
    else:
        image, label = read_and_decode_xent(filename_queue)

    min_after_dequeue = 50
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle is True:
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        # dont shuffle batches for validation
        images_batch, labels_batch = tf.train.batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True)
    return images_batch, labels_batch


def input_pipeline_miou(filenames, batch_size, fmt, num_epochs=None, shuffle=True):
    """The input pipeline when using approximate iou loss

    Keyword arguments:
    filenames -- name of queue to read from
    batch_size -- the mini-batch size
    num_epochs -- max number of epochs
    shuffle -- should data be shuffled
    """
    filename_queue = tf.train.string_input_producer(
        [filenames], num_epochs=num_epochs, shuffle=shuffle)

    if fmt == 'rgb':
        image, label = read_and_decode_miou(filename_queue)
    else:
        image, label = read_and_decode_lab(filename_queue)

    min_after_dequeue = 50
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle is True:
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        # dont shuffle batches for validation
        images_batch, labels_batch = tf.train.batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True)
    return images_batch, labels_batch


def init_4subplot():

    dummy_image1 = np.random.uniform(0, 255, size=(3, 224, 224))
    dummy_image2 = np.random.uniform(0, 255, size=(1, 224, 224))

    fig = plt.figure()

    a = fig.add_subplot(1, 4, 1)
    b = fig.add_subplot(1, 4, 2)
    c = fig.add_subplot(1, 4, 3)
    d = fig.add_subplot(1, 4, 4)

    a.set_axis_off()
    b.set_axis_off()
    c.set_axis_off()
    d.set_axis_off()

    img_1 = a.imshow(dummy_image1[0])
    img_2 = b.imshow(dummy_image2[0])
    img_3 = c.imshow(dummy_image2[0])
    img_4 = d.imshow(dummy_image2[0])

    return img_1, img_2, img_3, img_4


def init_3subplot():

    dummy_image1 = np.random.uniform(0, 255, size=(3, 224, 224))
    dummy_image2 = np.random.uniform(0, 255, size=(1, 224, 224))

    fig = plt.figure()

    a = fig.add_subplot(1, 3, 1)
    b = fig.add_subplot(1, 3, 2)
    c = fig.add_subplot(1, 3, 3)

    a.set_axis_off()
    b.set_axis_off()
    c.set_axis_off()

    img_1 = a.imshow(dummy_image1[0])
    img_2 = b.imshow(dummy_image2[0])
    img_3 = c.imshow(dummy_image2[0])

    return img_1, img_2, img_3


def update_plots(img_1, img_2, img_3, rgb, predimg, label, batch_size, n_classes):

    # The RGB
    img1 = np.asarray(rgb[0, :, :, :])
    img1 = np.squeeze(rgb[0, :, :, :])
    img_1.set_data(img1)

    # The prediction
    predimg = predimg.reshape(batch_size, 224, 224, 1)
    img2 = np.asarray(predimg[0, :, :, :])
    img2 = np.squeeze(predimg[0, :, :, :])
    img_2.set_data(img2 * 255. / n_classes)

    # The mask
    lbl = np.asarray(label[0, :, :, :])
    lbl = np.squeeze(lbl)
    img_3.set_data(lbl * 255. / n_classes)


def get_minibatch(batch_size, lumi_list, alph_list, beta_list, mask_list):

    minibatch = np.zeros([batch_size, 224, 224, 4])

    for i in range(batch_size):

        j = random.randint(0, len(lumi_list) - batch_size)

        lumi = Image.open(lumi_list[j])
        alph = Image.open(alph_list[j])
        beta = Image.open(beta_list[j])
        mask = plt.imread(mask_list[j])

        mask.resize((224, 224))

        minibatch[i, :, :, 0] = lumi.resize((224, 224))
        minibatch[i, :, :, 1] = alph.resize((224, 224))
        minibatch[i, :, :, 2] = beta.resize((224, 224))
        minibatch[i, :, :, 3] = mask.copy()

    return minibatch


def email_results(step, max_valid):

    to_address = 'gallowaa@mail.uoguelph.ca'

    # Send email with results
    machine_name = socket.gethostname().split('.')[0]
    session_name = os.environ['STY'].split('.')[1]

    # Create a file called .env with the following contents:
    ''' 
    USERNAME:<username@gmail.com>
    PASSWORD:<password>
    '''
    USERNAME = 0
    PASSWORD = 1

    fp = open('.env', 'r')
    ENV = fp.read()
    ENV_TOKENS = ENV.split('\n')
    username = ENV_TOKENS[USERNAME].split(':')[1]
    password = ENV_TOKENS[PASSWORD].split(':')[1]
    fp.close()

    msg = MIMEText('Ran for %d steps \n\n modify body as needed' % step)

    msg['Subject'] = '%s @ %s finished with %.4f' % (
        session_name, machine_name, max_valid)
    msg['From'] = username
    msg['To'] = to_address

    # Send the message via GMAIL SMTP server, but don't include the
    # envelope header.
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(username, [to_address], msg.as_string())
    server.quit()
