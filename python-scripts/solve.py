from __future__ import division
import sys
import os
import caffe
import numpy as np
import subprocess
from py_img_seg_eval.eval_segm import *
# base net -- the learned coarser model
# base_weights = './RAODNN.caffemodel' # https://gist.github.com/longjon/1bf3aa1e0b8e788d7e1d
solver = caffe.SGDSolver('solver.prototxt')

# copy base weights for fine-tuning
# solver.net.copy_from(base_weights)
# solver.net.set_mode_gpu()
# solver.net.set_device(0)
# print [(k, v[0].data, v[1].data) for k, v in solver.net.params.items()]
## control layer's initialization
halt_training = False
for layer in solver.net.params.keys():
  for index in range(0, 2):
    if len(solver.net.params[layer]) < index+1:
      continue

    if np.sum(solver.net.params[layer][index].data) == 0:
      print layer + ' is composed of zeros!'
      halt_training = True

if halt_training:
  print 'Exiting.'
  exit()


test_interval = 1000

FNULL = open(os.devnull, 'w')
for i in xrange(100000):
    solver.step(1)

    if i > 0 and (i % test_interval) == 0:
      subprocess.call(['python', 'test.py', str(i)], stderr=FNULL)