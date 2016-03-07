import theano
import numpy as np
import theano.tensor.signal.conv as conv
import theano.tensor.nnet.conv


class CNNlayer(object):
    #input and filter shapes are 4-vectors, (number of inputs/filters, channels, height, width)
    def __init__(self, input_shape, filter_shape):
        assert input_shape[1] == filter_shape[1]
        self.batch_size   = input_shape[0]
        self.num_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width  = input_shape[3]
        self.num_filters  = filter_shape[0]
        self.filter_height= filter_shape[2]
        self.filter_width = filter_shape[3]
        assert self.filter_height % 2 == 1
        assert self.filter_width % 2 == 1
        self.W = theano.shared(np.random.rand(self.num_channels,self.num_filters, self.filter_height, self.filter_width))
        self.b = theano.shared(np.random.rand(self.num_filters))
        
        
    def setW(self, w):
        self.W = w

    def updateW(self, w):
        self.W =+ w

    def pad(self, input):
        if self.filter_height == 1 and self.filter_width ==1 :
            return
        else:
            w = int(self.filter_width/2)
            h = int(self.filter_height/2)
            input = np.lib.pad(input, ((0,0), (0,0), (h, h), (w,w)), mode='constant', constant_values=(0))
            return input

    def manual(self, inp):
        assert len(inp) == self.batch_size
        assert len(inp[0]) == self.num_channels
        assert len(inp[0][0]) == self.input_height
        assert len(inp[0][0][0]) == self.input_width
        fh = self.filter_height
        fw = self.filter_width
        ih = self.input_height
        iw = self.input_width
        input = self.pad(inp)
        sections = np.ones((self.batch_size, self.num_channels, ih, iw, fh, fw))
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                for input_row in range(ih):
                    for input_col in range(iw):
                        for ker_row in range(fh):
                            for ker_col in range(fw):
                                sections[b][c][input_row][input_col][ker_row][ker_col]= input[b][c][ker_row + input_row][ker_col + input_col]
                                
        summands = np.ones((self.batch_size, self.num_channels, ih, iw, self.num_filters))
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                for h in range(ih):
                    for w in range(iw):
                        for f in range(self.num_filters):
                            temp = sections[b][c][h][w] * self.W[f][c] + self.b[f]
                            summands[b][c][h][w][f]=np.sum(temp.eval())
        #results = theano.tensor.nnet.sigmoid(summands)
        return input, summands

    def eval(self, inp):
        #input = self.pad(inp.eval())
        results= theano.tensor.nnet.conv2d(input, self.W, border_mode='full' )
        biased = results + self.b.dimshuffle('x', 0, 'x', 'x')
        result = theano.tensor.nnet.sigmoid(biased)
        return result

    
    




#test = CNNlayer((1,1,4,4), (1,1,3,3))
#inp = np.array([[0.0,0,0,0],[0,1,0,0], [0,0,0,0], [0,0,0,0]])
#weight = np.array([[1,.2, 0],[.4,.5, 0], [0,0,0]])
#test.setW(weight)
#weight_2 = np.array([list(weight[i][::-1]) for i in range(len(weight))])
#weight_3 = weight_2[::-1]
#print(inp)
#print(( weight_3))
#print("their", conv.conv2d(inp, weight).eval())


test = CNNlayer((1,1,4,3), (1,1,3,3))

arr = np.array([[[[1.0,2,3], [4,5,6],[7,8,9], [10,11,12]]]])
input = theano.tensor.tensor4(name="input")
evaluate = theano.function([input], test.eval(input))
#print(test.W)
#print(evaluate(arr))
input2 = theano.tensor.tensor4(name="input2")
cost = ((evaluate(input2)**2).sum())
input3 = theano.tensor.scalar(name="input3")
gradient = theano.tensor.grad(cost,test.W)
print(gradient(cost))
