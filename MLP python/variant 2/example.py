#!/usr/bin/python

import pylab as pl
import numpy as np
import ANN

# target inputs (Example)
inputs = [np.array([ 5.0, 2.2, 3.1]),
          np.array([ 4.8, 1.9, 3.0]),
          np.array([-8.0,-2.0,-2.9]),
          np.array([-7.8,-2.1,-2.7])]

# target outputs (Example)
targets = [np.array([1,0]),
           np.array([1,0]),
           np.array([0,1]),
           np.array([0,1])]

# layers (Example)
LayerSizes = [3,44,88,2]

NN = ANN.ANN(LayerSizes, -1.0, 2.0, 0.125)

Err = NN.train(inputs, targets, 80)

Res = NN.forward(inputs[0])
print "input 0 gives [%2.2f, %2.2f] " % (Res[0], Res[1])
Res = NN.forward(inputs[1])
print "input 1 gives [%2.2f, %2.2f] " % (Res[0], Res[1])
Res = NN.forward(inputs[2])
print "input 2 gives [%2.2f, %2.2f] " % (Res[0], Res[1])
Res = NN.forward(inputs[3])
print "input 3 gives [%2.2f, %2.2f] " % (Res[0], Res[1])

ExInp = np.random.rand(3)*10 - 5
Res = NN.forward(ExInp)
print "random input gives [%2.2f, %2.2f] " % (Res[0], Res[1])

fig = pl.plot(Err)
pl.title("Network Error")
pl.show()
