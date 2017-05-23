import os
import re
import random
import time
import os.path
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from natsort import natsorted
from pprint import pprint
from PIL import Image


def read_filelist(img_path):

    # Q for luminance
    lumList = [os.path.join(dirpath, f)
               for dirpath, dirnames, files in os.walk(img_path + 'lum-float/')
               for f in files if f.endswith('.tiff')]
    lumList = natsorted(lumList)
    print("No of lum files: %i" % len(lumList))
    #lumFiles = tf.train.string_input_producer(lumList, shuffle=False)

    # Q for alpha
    alphaList = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(img_path + 'alpha-float/')
                 for f in files if f.endswith('.tiff')]
    alphaList = natsorted(alphaList)
    print("No of alpha files: %i" % len(alphaList))
    #alphaFiles = tf.train.string_input_producer(alphaList, shuffle=False)

    # Q for beta
    betaList = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(img_path + 'beta-float/')
                for f in files if f.endswith('.tiff')]
    betaList = natsorted(betaList)
    print("No of beta files: %i" % len(betaList))
    #betaFiles = tf.train.string_input_producer(betaList, shuffle=False)

    # Q for segmentations - still jpeg
    segList = [os.path.join(dirpath, f)
               for dirpath, dirnames, files in os.walk(img_path + 'segmentations/')
               for f in files if f.endswith('.jpg')]
    segList = natsorted(segList)
    print("No of mask files: %i" % len(segList))
    #segFiles = tf.train.string_input_producer(segList, shuffle=False)

    #return lumFiles, len(lumList), alphaFiles, len(alphaList), betaFiles, len(betaList), segFiles, len(segList)
    return lumList, alphaList, betaList, segList
    # return
    # alphaFiles,len(alphaList),betaFiles,len(betaList),segFiles,len(segList)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == '__main__':

    BYTE = 8
    FLOAT = 32
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', default=FLOAT)
    parser.add_argument('--dataset_kind', default="val")
    parser.add_argument('--root_path', help="path to images",
                        default="/export/mlrg/gallowaa/Documents/ciona-net/data/")
    parser.add_argument('--out_path', help="tfrecord file name to write",
                        default="/scratch/gallowaa/tfrecord/")
    parser.add_argument(
        '--crop', help="should we crop/pad images to fixed size?", default="y")
    args = parser.parse_args()

    if args.precision == BYTE:
        datatype = 'uint8'
    else:
        datatype = 'float32'

    if args.dataset_kind == 'train':
        images_path = os.path.join(args.root_path, 'data-train-ciona-16/')
        filename = os.path.join(
            args.out_path, 'ciona-16-lab-' + datatype + '-702.tfrecords')
    else:
        images_path = os.path.join(args.root_path, 'data-val-ciona-16/')
        filename = os.path.join(
            args.out_path, 'ciona-16-lab-' + datatype + '-334.tfrecords')

    # read file list from directory into queues
    #lQ, lLen, aQ, aLen, bQ, bLen, sQ, segLen = read_filelist(images_path)

    lumList, alphaList, betaList, segList = read_filelist(images_path)

    '''
    reader = tf.WholeFileReader()
    _, lvalue = reader.read(lQ)
    _, avalue = reader.read(aQ)
    _, bvalue = reader.read(bQ)
    _, mvalue = reader.read(sQ)

    lumi = tf.image.decode_png(lvalue)
    alph = tf.image.decode_png(avalue)
    beta = tf.image.decode_png(bvalue)
    mask = tf.image.decode_jpeg(mvalue)
    '''

    '''
    if args.crop == "y":
        print('Resizing images to 224x224')
        lumi = tf.image.resize_image_with_crop_or_pad(lumi, 224, 224)
        alph = tf.image.resize_image_with_crop_or_pad(alph, 224, 224)
        beta = tf.image.resize_image_with_crop_or_pad(beta, 224, 224)
        mask = tf.image.resize_image_with_crop_or_pad(mask, 224, 224)
    '''

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # filename=os.path.join(args.rec+'.tfrecords')

        print('Writing', filename)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.python_io.TFRecordWriter(filename)

        for i in tqdm(range(len(lumList))):  # (11648)

            lumiRaw = Image.open(lumList[i])
            alphRaw = Image.open(alphaList[i])
            betaRaw = Image.open(betaList[i])
            maskRaw = plt.imread(segList[i])

            '''
            lumi_arr = lumi.eval()
            alph_arr = alph.eval()
            beta_arr = beta.eval()
            mask_arr = mask.eval()
            lumiRaw = lumi_arr.tostring()
            alphRaw = alph_arr.tostring()
            betaRaw = beta_arr.tostring()
            maskRaw = mask_arr.tostring()
            '''

            if args.precision == BYTE:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'lumi_raw': _bytes_feature(lumiRaw),
                    'alph_raw': _bytes_feature(alphRaw),
                    'beta_raw': _bytes_feature(betaRaw),
                    'mask_raw': _bytes_feature(maskRaw)}))
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'lumi_raw': _float_feature(list(lumiRaw.getdata())),
                    'alph_raw': _float_feature(list(alphRaw.getdata())),
                    'beta_raw': _float_feature(list(betaRaw.getdata())),
                    'mask_raw': _bytes_feature(maskRaw.tobytes())
                    }))

                writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)
