import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def conv(bottom, nout, ks=1, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return  conv

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def crf(split):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=161.482 ,#, 162.052, 175.939),
            seed=1337)

    n.data, n.label = L.Python(module='road_layer', layer='ROADSegDataLayer',
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 32, ks=5)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 32, ks=5)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 64)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 64)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 128)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 128)
    n.pool3 = max_pool(n.relu3_2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 256)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 256)
    # n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 256)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 256)
    # n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_2)

    n.score = conv(n.pool5, 1)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    # n.loss = L.MultiStageMeanfield()
    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(crf('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(crf('valid')))

if __name__ == '__main__':
    make_net()