caffe_root = '../CRFasRNN-merge/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import os
import caffe
import random
from defs import CACHE_PATH

class ROADSegDataLayer(caffe.Layer):
    """
    Load (image,label) from 'CACHE' one-at-a-time
    while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - batch_size
        - split: train / val / test
        - randomize: load in random order
        - seed: seed for randomization

        for ROAD crack identification.

        example

        params = dict(voc_dir="/home/public/data/cache",
            mean=(161.482, 162.052, 175.939),
            split="val")

        :param bottom:
        :param top:
        :return:
        """
        # config
        params = eval(self.param_str)

        self.split = params['split']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batchsize = params.get('batchsize', 1)
        print self.batchsize

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        if self.split == 'test':
            self.x = np.load(os.path.join(CACHE_PATH, 'x_test.npy'))
            self.y = np.load(os.path.join(CACHE_PATH, 'y_test.npy'))
        else:
            self.x = np.load(os.path.join(CACHE_PATH, 'x_{}_GA06A.npy'.format(self.split)))
            self.y = np.load(os.path.join(CACHE_PATH, 'y_{}_GA06A.npy'.format(self.split)))

        self.x = self.x.astype('float32')
        self.y = self.y.astype('uint8')
        mean = np.mean(self.x)
        self.x -= mean

        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, np.shape(self.x)[0]-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.x[self.idx]
        self.label = self.y[self.idx]
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batchsize, *self.data.shape)
        top[1].reshape(self.batchsize, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, np.shape(self.x)[0] - 1)
        else:
            self.idx += 1
            if self.idx == np.shape(self.x)[0]:
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

