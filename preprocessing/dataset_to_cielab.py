import os
import os.path
import argparse
from natsort import natsorted

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import data, img_as_float, exposure, color


def read_filelist(img_path):

    imgList = [os.path.join(dirpath, f)
               for dirpath, dirnames, files in os.walk(img_path)
               for f in files if f.endswith('.jpg')]
    imgList = natsorted(imgList)
    print "No of files: %i" % len(imgList)

    return imgList, len(imgList)

if __name__ == '__main__':

    rescale = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_kind', default="val")
    parser.add_argument('--root_path', help="path to images",
                        default="/home/angus/Documents/ciona-net/data/")

    args = parser.parse_args()

    if args.dataset_kind == 'train':
        base_path = os.path.join(args.root_path, 'data-train-ciona-16/')
        images_path = os.path.join(base_path, 'rgb/')
        segmentations_path = os.path.join(base_path, 'segmentation/')
    else:
        base_path = os.path.join(args.root_path, 'data-val-ciona-16/')
        images_path = os.path.join(base_path, 'rgb/')
        segmentations_path = os.path.join(args.root_path, 'segmentation/')

    # read file list from directory into queues
    imgFiles, imgLen = read_filelist(images_path)

    #for i in range(2):
    for i in tqdm(range(imgLen)):

        rgb = plt.imread(imgFiles[i])
        lab = color.rgb2lab(rgb)
        lab[:, :, 0] *= 255. / 100
        lab[:, :, 1] += 128
        lab[:, :, 2] += 128

        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        if rescale == True:
            # .astype(np.uint8)
            L = (((L - L.min()) / (L.max() - L.min())) * 255)
            # .astype(np.uint8)
            a = (((a - a.min()) / (a.max() - a.min())) * 255)
            # .astype(np.uint8)
            b = (((b - b.min()) / (b.max() - b.min())) * 255)
        
        else:
            L = L.astype(np.uint8)
            a = a.astype(np.uint8)
            b = b.astype(np.uint8)

        img_L = Image.fromarray(L)
        img_a = Image.fromarray(a)
        img_b = Image.fromarray(b)

        if rescale == True:
            img_L.save(base_path + 'lum-float/' + 
                os.path.basename(imgFiles[i])[:-4] + '.tiff')
            img_a.save(base_path + 'alpha-float/' +
                       os.path.basename(imgFiles[i])[:-4] + '.tiff')
            img_b.save(base_path + 'beta-float/' +
                       os.path.basename(imgFiles[i])[:-4] + '.tiff')
        '''
        else:
            img_L.save(base_path + 'lum/' +
                       imgFiles[i][PATH_LEN:-4] + '.png')
            img_a.save(base_path + 'alpha/' +
                       imgFiles[i][PATH_LEN:-4] + '.png')
            img_b.save(base_path + 'beta/' +
                       imgFiles[i][PATH_LEN:-4] + '.png')
        '''
