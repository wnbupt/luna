import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import Image

# Make sure that caffe is on the python path:
caffe_root = '../CRFasRNN-merge/caffe-master/' # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the net, list its data and params, and filter an example image.
caffe.set_mode_cpu()
net = caffe.Net('./net_surgery/conv.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

# load image and prepare as a single input batch for Caffe
im = np.array(Image.open('./images/cat_gray.jpg'))
plt.title("original image")
plt.imshow(im)
plt.axis('off')

im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input