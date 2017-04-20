#! /bin/sh

# TOOLS=../CRFasRNN-merge/caffe-master/build/tools
TOOLS=../caffe-master/build/tools
MODEL=test.prototxt #solver.prototxt
WEIGHTS=./snapshot/train_iter_50000.caffemodel

$TOOLS/caffe test -model $MODEL -weights $WEIGHTS -gpu 0 -iterations 10000