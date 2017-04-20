import caffe
import numpy as np

smooth = 1.
#np.set_printoptions(threshold='nan')

class DiceCoefLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance")

    def reshape(self, bottom, top):
        if bottom[0].num != bottom[1].num:
            raise Exception("Input# difference is shape of inputss must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.ones_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        pred = bottom[0].data.flatten()
        truth = bottom[1].data.flatten()
        intersection = np.sum(truth * pred)
        diff = (2. * intersection + smooth) / (np.sum(truth) + np.sum(pred) + smooth)
        # self.diff *= diff
        self.diff = diff*(bottom[1].data-bottom[0].data)
        # print bottom[0].data
        top[0].data[...] = 1-diff

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = -self.diff #/ bottom[i].num


