import numpy as np
import os
import os.path
import tensorflow as tf
from tqdm import tqdm
from natsort import natsorted
import random
import time
import re
from pprint import pprint
import argparse


def read_filelist(img_path, seg_path):

    imgList = [os.path.join(dirpath, f)
               for dirpath, dirnames, files in os.walk(img_path)
               for f in files if f.endswith('.jpg')]
    imgList = natsorted(imgList)
    print("No of files: %i" % len(imgList))
    imgFiles = tf.train.string_input_producer(imgList, shuffle=False)

    segList = [os.path.join(dirpath, f)
               for dirpath, dirnames, files in os.walk(seg_path)
               for f in files if f.endswith('.jpg')]
    segList = natsorted(segList)
    print("No of files: %i" % len(segList))
    segFiles = tf.train.string_input_producer(segList, shuffle=False)

    return imgFiles, len(imgList), segFiles, len(segList)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', help="root path",
                        default="~/ciona17_farm1_training1/")
    parser.add_argument('--img', help="path to images",
                        default="images/")
    parser.add_argument('--seg', help="path to segmentations",
                        default="masks/")
    parser.add_argument('--out_path', help="tfrecord file name to write",
                        default="~/tfrecord/")
    parser.add_argument(
        '--rec', help="tfrecord file name to write", default="ciona-17-rgb-702")
    parser.add_argument(
        '--crop', help="should we crop/pad images to fixed size?", default="y")
    args = parser.parse_args()

    # read file list from directory into queues
    imgpath = os.path.join(args.in_path, args.img)
    segpath = os.path.join(args.in_path, args.seg)

    print('Will load images from %s' % imgpath)
    print('Will load masks from %s' % segpath)

    iQ, imgLen, mQ, segLen = read_filelist(imgpath, segpath)

    reader = tf.WholeFileReader()
    key, ivalue = reader.read(iQ)
    key, mvalue = reader.read(mQ)
    myImg = tf.image.decode_jpeg(ivalue)  # reads images into tensor
    mySeg = tf.image.decode_jpeg(mvalue)  # reads images into tensor

    if args.crop == "y":
        print('Resizing images to 224x224')
        myImg = tf.image.resize_image_with_crop_or_pad(myImg, 224, 224)
        mySeg = tf.image.resize_image_with_crop_or_pad(mySeg, 224, 224)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        filename = os.path.join(args.out_path, args.rec) + '.tfrecords'
        print('Writing', filename)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.python_io.TFRecordWriter(filename)

        for index in tqdm(range(imgLen)):  # (11648)
            image = myImg.eval()
            mask = mySeg.eval()
            imageRaw = image.tostring()
            maskRaw = mask.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(imageRaw),
                'mask_raw': _bytes_feature(maskRaw)}))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)
