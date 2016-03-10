import theano
import theano.tensor as T
import lasagne
import lasagne.nonlinearities as nl
import numpy as np

#Parameters
glimpse_size = 20
glimpse_elements = glimpse_size**2
num_glimpses = 10

glimpse_number_of_convolving_filters = 32
glimpse_convolving_filter_size = 9          #must be odd
glimpse_output_size = 20

recurrent_output_size = 40

classification_units = 20

downsample_rows = 100
downsample_cols = 100

context_number_of_convolving_filters = 64
context_convolving_filter_size = 19          #must be odd
context_pool_rate = 2

#Glimpse Variables
conv1_weights = theano.shared (np.random.normal (0.0, 0.01, 
              (glimpse_number_of_convolving_filters, glimpse_elements,
               glimpse_convolving_filter_size, glimpse_convolving_filter_size)))
conv2_weights = theano.shared (np.random.normal (0.0, 0.01, 
              (glimpse_number_of_convolving_filters, glimpse_elements, 
               glimpse_convolving_filter_size, glimpse_convolving_filter_size)))
conv3_weights = theano.shared (np.random.normal (0.0, 0.01, 
              (glimpse_number_of_convolving_filters, glimpse_elements, 
               glimpse_convolving_filter_size, glimpse_convolving_filter_size)))
what_weights  = theano.shared (np.random.normal (0.0, 0.01, 
              (glimpse_elements* glimpse_number_of_convolving_filters, glimpse_output_size)))
where_weights = theano.shared (np.random.normal (0.0, 0.01, 
              (glimpse_elements* glimpse_number_of_convolving_filters, glimpse_output_size)))
conv1_bias = theano.shared(np.random.normal(0.0, 0.01, glimpse_number_of_convolving_filters))
conv2_bias = theano.shared(np.random.normal(0.0, 0.01, glimpse_number_of_convolving_filters))
conv3_bias = theano.shared(np.random.normal(0.0, 0.01, glimpse_number_of_convolving_filters))
what_bias  = theano.shared(np.random.normal(0.0, 0.01, glimpse_output_size))
where_bias = theano.shared(np.random.normal(0.0, 0.01, glimpse_output_size))


#Recurrent Variables
W1 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]
W2 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]
U1 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]
U2 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]
Vo1 = theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size))
Vo2 = theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size))
b1 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]
b2 = [theano.shared (np.random.normal (0.0, 0.01, glimpse_output_size)) for i in range(4)]

#Emission Variables
emission_weights = theano.shared(np.random.normal(0.0, 0.01, (2, glimpse_output_size)))
emission_bias = theano.shared(np.random.normal( 0.0, 0.01, 2))

#Classification Variables
class_weights = theano.shared(np.random.normal(0.0, 0.01, (glimpse_output_size, classification_units)))
class_bias = theano.shared(np.random.normal( 0.0, 0.01, classification_units))



class GlimpseLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        batch, location = input
        rows = batch.shape[2]-glimpse_size
        cols = batch.shape[3]-glimpse_size
        start_row = location[0]*rows
        start_col =location[1]*cols
        return A[:, :, start_row:start_row+glimpse_size, start_col:start_col+glimpse_size] 

    def get_output_shape_for(self, input_shape):
        return [input_shape[0], input_shape[1], glimpse_size, glimpse_size]

    #input of 16 arguments
    #ouput of 2 arguments
class CustomLSTM(lasagne.layers.Layer):
    def __init__(self, incoming, W, U, Vo, b, **kwargs):
        self.Wi = W[0]
        self.Wf = W[1]
        self.Wc = W[2]
        self.Wo = W[3]
        self.Ui = U[0]
        self.Uf = U[1]
        self.Uc = U[2]
        self.Uo = U[3]
        self.Vo = Vo
        self.bi = b[0]
        self.bf = b[1]
        self.bc = b[2]
        self.bo = b[3]
        super().__init__(incoming, **kwargs)

    def get_output_for(self, arguments, **kwargs):
        input, hprev, Cprev = arguments
        i = nl.sigmoid(self.Wi * input + self.Ui* hprev + self.bi)
        cand = nl.tanh(self.Wc *input + self.Uc * hprev + self.bc)
        f = nl.sigmoid(self.Wf*input + self.Uf*hprev + self.bf)
        C = i*cand + f * Cprev
        o = nl.sigmoid(self.Wo*input + self.Uo*hprev + self.Vo*C + self.bo)
        h = o*nl.tanh(C)
        return h, C

    def get_output_shape_for(self, input_shape):
        return (2, glimpse_output_size)

#inputs are tuple of glimpse (size glimpse_size by glimpse_size) and location tuple
#output is Gn of length glimpse_output_size
def build_glimpse_network(glimpse = None, location = None):
    if not isinstance(glimpse, lasagne.layers.Layer):
        glimpse_in = lasagne.layers.InputLayer((None, 1, glimpse_size, glimpse_size), glimpse)
    else:
        glimpse_in = glimpse
    if not isinstance(location, lasagne.layers.Layer):
        location_in = lasagne.layers.InputLayer((None, 1, 1, 2), location)
    else:
        location_in = location
    glimpse_in = lasagne.layers.InputLayer(shape=(None, 1, glimpse_size, glimpse_size),
                                     input_var = glimpse)
    location_in = lasagne.layers.InputLayer(shape=(None, 1, 1, 2), input_var=location)
    first_conv = lasagne.layers.Conv2DLayer(glimpse_in,
                                             glimpse_number_of_convolving_filters,
                                             glimpse_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify, 
                                             W = conv1_weights, 
                                             b = conv1_bias)
    second_conv = lasagne.layers.Conv2DLayer(first_conv,
                                             glimpse_number_of_convolving_filters,
                                             glimpse_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify, 
                                             W = conv2_weights, 
                                             b = conv2_bias)
    third_conv = lasagne.layers.Conv2DLayer(second_conv,
                                             glimpse_number_of_convolving_filters,
                                             glimpse_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify, 
                                             W = conv3_weights, 
                                             b = conv3_bias)
    what = lasagne.layers.DenseLayer(third_conv, 
                                             num_units = glimpse_output_size, 
                                             nonlinearity = nl.rectify, 
                                             W = what_weights, 
                                             b = what_bias)
    where = lasagne.layers.DenseLayer(location_in, 
                                             num_units = glimpse_output_size, 
                                             nonlinearity = nl.rectify, 
                                             W = where_weights,
                                             b = where_bias)
    Gn = lasagne.layers.ElemwiseMergeLayer((what, where), merge_function='mul')
    return Gn

#input is Gn of length glimpse_output_size
#output is r1 and r2 of length glimpse_output_size
def build_recurrent_network(Gn):
    if not isinstance(Gn, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size), Gn)
    else:
        l_in = Gn
    #r1 = lasagne.layers.LSTMLayer(l_in, num_units=recurrent_output_size, 
    #                              nonlinearity = None, 
    #                              peepholes=True, 
    #                              gradient_steps=-1,
    #                              unroll_scan = True)
    #r2 = lasagne.layers.LSTMLayer(r1, num_units=recurrent_output_size, 
    #                              nonlinearity = None, 
    #                              peepholes=True, 
    #                              gradient_steps=-1,
    #                              unroll_scan = True)

    r1 = CustomLSTM(l_in, W1, U1, Vo1, b1)
    r2 = CustomLSTM(l_in, W2, U2, Vo2, b2)
    return r1, r2

#input is r2 of length glimpse_output_size
#output is location, a tuple of floats
def build_emission_network(r2):
    if not isinstance(r2, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size, recurrent_output_size), r2)
    else:
        l_in = r2
    output = lasagne.layers.DenseLayer(l_in, 2, nonlinearity=nl.rectify, 
                                       W = emission_weights, b = emission_bias)
    return output

#input is r1 of length glimpse_output_size
#output is labels of length classification_units
def build_classification_network(r1):
    if not isinstance(r1, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size, recurrent_output_size), r1)
    else:
        l_in = r1
    output = lasagne.layers.DenseLayer(l_in, classification_units, 
                                       nonlinearity = nl.softmax, 
                                       W = class_weights, b = class_bias)
    return output

#input is downsampled batch of images
#output is initial r2, of length glimpse_output_size
def build_context_network(downsample):
    if not isinstance(downsample, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, 1, downsample_rows, downsample_cols), downsample)
    else:
        l_in = downsample
    first_conv = lasagne.layers.Conv2DLayer(l_in,
                                             context_number_of_convolving_filters,
                                             context_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify)
    first_pool = lasagne.layers.MaxPool2DLayer(first_conv, context_pool_rate)
    second_conv = lasagne.layers.Conv2DLayer(first_pool,
                                             context_number_of_convolving_filters,
                                             context_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify)
    second_pool = lasagne.layers.MaxPool2DLayer(second_conv, context_pool_rate)
    third_conv = lasagne.layers.Conv2DLayer(second_pool,
                                             context_number_of_convolving_filters,
                                             context_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = nl.rectify)
    third_pool = lasagne.layers.MaxPool2DLayer(third_conv, context_pool_rate)
    fc = lasagne.layers.DenseLayer(third_pool, 
                                       glimpse_output_size*recurrent_output_size, 
                                       nonlinearity = nl.rectify)
    output = lasagne.layers.ReshapeLayer(fc, (-1, glimpse_output_size, recurrent_output_size))
    return output


glimpse = T.tensor4('glimpse')
location = T.tensor4('location')
Gn = build_glimpse_network(glimpse, location)
#Gn is of shape (None, glimpse_output_size)
#print(lasagne.layers.get_output_shape(Gn, {'glimpse':[100,1,28,28], 'location':[100, 1, 1, 2]}))
r1, r2 = build_recurrent_network(Gn)
#Both r1 and r2 are of shape (None, glimpse_output_size, recurrent_output_size)
print(lasagne.layers.get_output_shape(r1, {'glimpse':[100,1,28,28], 'location':[100, 1, 1, 2]}))
print(lasagne.layers.get_output_shape(r2, {'glimpse':[100,1,28,28], 'location':[100, 1, 1, 2]}))
r2_init = build_context_network(glimpse)
print(lasagne.layers.get_output_shape(r2_init, {'glimpse':[100,1,28,28]}))
input = T.tensor4('input')




def main():
    downsample = T.tensor4('downsample')
    batch = T.tensor4('batch')
    r2 = build_context_network(downsample)
    loc = build_emission_network(r2_init)
    glimpse = GlimpseLayer((batch, loc_init))
    
    for i in range(num_glimpses):
        Gn = build_glimpse_network(glimpse, loc)
        r1, r2 = build_recurrent_network(Gn)
        loc = build_emission_network(r2_init)


