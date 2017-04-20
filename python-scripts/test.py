import os, sys
import numpy as np
import caffe
from defs import CACHE_PATH
from py_img_seg_eval.eval_segm import *

def main():
  iteration_num = process_arguments(sys.argv)

  prototxt = 'test.prototxt'
  caffemodel = 'models/train_iter_{}.caffemodel'
  # class_names = ['bird', 'bottle', 'chair']
  # class_ids = get_id_classes(class_names)
  # lut = create_lut(class_ids)


  # images, labels = create_full_paths(file_names, 'images', 'labels')

  test_net(prototxt, caffemodel.format(iteration_num))



def test_net(prototxt, caffemodel, images, labels, lut):
  net = caffe.Segmenter(prototxt, caffemodel, True)

  pa_list    = []
  ma_list    = []
  m_IU_list  = []
  fw_IU_list = []

  images = np.load(os.path.join(CACHE_PATH, 'x_test.npy'))
  labels = np.load(os.path.join(CACHE_PATH, 'y_test.npy'))
  images = images.astype('float32')
  labels = labels.astype('uint8')
  mean = np.mean(images)
  images -= mean
  for idx in xrange(0, images.shape[0]):
    pred = net.predict(images[idx])
    ba = pixel_accuracy(pred, labels[idx])
    ma = mean_accuracy(pred, labels[idx])
    m_IU = mean_IU(pred, labels[idx])
    fw_IU = frequency_weighted_IU(pred, labels[idx])

    pa_list.append(ba)
    ma_list.append(ma)
    m_IU_list.append(m_IU)
    fw_IU_list.append(fw_IU)

  print("pixel_accuracy: " + str(np.mean(pa_list)))
  print("mean_accuracy: " + str(np.mean(ma_list)))
  print("mean_IU: " + str(np.mean(m_IU_list)))
  print("frequency_weighted: " + str(np.mean(fw_IU_list)))

def process_arguments(argv):
  if len(argv) != 2:
    help()
  else:
    iteration_num = argv[1]

  return iteration_num


if __name__ == "__main__":
    main()

