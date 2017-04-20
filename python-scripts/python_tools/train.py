import _init_paths
import caffe
import os
ROOT_DIR = os.getcwd()
solver_prototxt = os.path.join(ROOT_DIR,'..','solver.prototxt')
output_dir = os.path.join(ROOT_DIR,'..')
solver = caffe.SGDSolver(solver_prototxt)
max_iters=1000
while solver.iter < max_iters:
# Make one SGD update
    solver.step(1)
net = solver.net
filename = ('RAODNN' + '.caffemodel')
filename = os.path.join(output_dir, filename)
net.save(str(filename))