#! /bin/sh

TOOLS=/home/public/CRFasRNN/caffe-master/build/tools
WEIGHTS=ROAD.caffemodel
SOLVER=solver_adam.prototxt #solver.prototxt
SNAPSHOT=./snapshot/train/train_iter_1000.solverstate

$TOOLS/caffe train  -solver $SOLVER -gpu 0 #-snapshot $SNAPSHOT
