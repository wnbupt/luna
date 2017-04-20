# caffe_root = '../caffe/'
caffe_root = '../CRFasRNN-merge/caffe-master/'
import os, sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np

ROOT_DIR = os.getcwd()
solver_prototxt = os.path.join(ROOT_DIR, 'solver_adam.prototxt')
solver = caffe.SGDSolver(solver_prototxt)
max_iters = 1000
# Load the original network and extract the CNN layers' parameters.
conv_net = caffe.Net('./train_test.prototxt',
                './snapshot/train_iter_100000.caffemodel',
                caffe.TEST)
layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv4', 'conv5', 'conv6']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (conv_net.params[pr][0].data, conv_net.params[pr][1].data) for pr in layers}
params = {pr: (solver.net.params[pr][0].data, solver.net.params[pr][1].data) for pr in layers}

for layer in layers:
    params[layer][0][...] = conv_params[layer][0]
    params[layer][1][...] = conv_params[layer][1]

for conv in layers:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, params[conv][0].shape, params[conv][1].shape)

solver.net.save('surgery_conv.caffemodel')
solver.solve()

#filename = os.path.join(ROOT_DIR, 'RAODNN.caffemodel')
#solver.net.save(str(filename))

