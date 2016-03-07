import theano.tensor as T
import theano
import numpy as np

A = theano.shared(np.arange(81).reshape(3,3,3,3))
B = A[:, :, 0:2, 0:2] 

print("A", A.eval())
print("B", B.eval())
