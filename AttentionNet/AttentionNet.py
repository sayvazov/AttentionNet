import theano
import theano.tensor as T
import lasagne

glimpse_size = 20
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
                                             nonlinearity = lasagne.nonlinearities.rectify)
    second_conv = lasagne.layers.Conv2DLayer(first_conv,
                                             glimpse_number_of_convolving_filters,
                                             glimpse_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = lasagne.nonlinearities.rectify)
    third_conv = lasagne.layers.Conv2DLayer(second_conv,
                                             glimpse_number_of_convolving_filters,
                                             glimpse_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = lasagne.nonlinearities.rectify)
    what = lasagne.layers.DenseLayer(third_conv, 
                                             num_units = glimpse_output_size, 
                                             nonlinearity = lasagne.nonlinearities.rectify)
    where = lasagne.layers.DenseLayer(location_in, 
                                             num_units = glimpse_output_size, 
                                             nonlinearity = lasagne.nonlinearities.rectify)
    Gn = lasagne.layers.ElemwiseMergeLayer((what, where), merge_function='mul')
    return Gn


def build_recurrent_network(Gn):
    if not isinstance(Gn, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size), Gn)
    else:
        l_in = Gn
    r1 = lasagne.layers.LSTMLayer(l_in, num_units=recurrent_output_size, 
                                  nonlinearity = None, 
                                  peepholes=True, 
                                  gradient_steps=-1,
                                  unroll_scan = True)
    r2 = lasagne.layers.LSTMLayer(r1, num_units=recurrent_output_size, 
                                  nonlinearity = None, 
                                  peepholes=True, 
                                  gradient_steps=-1,
                                  unroll_scan = True)
    return r1, r2

def build_emission_network(r2):
    if not isinstance(r2, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size, recurrent_output_size), r2)
    else:
        l_in = r2
    output = lasagne.layers.DenseLayer(l_in, 2, nonlinearity=lasagne.nonlinearities.rectify)
    return output

def build_classification_network(r1):
    if not isinstance(r1, lasagne.layers.Layer):
        l_in = lasagne.layers.InputLayer((None, glimpse_output_size, recurrent_output_size), r1)
    else:
        l_in = r1
    output = lasagne.layers.DenseLayer(l_in, classification_units, 
                                       nonlinearity = lasagne.nonlinearities.softmax)
    return output

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
                                             nonlinearity = lasagne.nonlinearities.rectify)
    first_pool = lasagne.layers.MaxPool2DLayer(first_conv, context_pool_rate)
    second_conv = lasagne.layers.Conv2DLayer(first_pool,
                                             context_number_of_convolving_filters,
                                             context_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = lasagne.nonlinearities.rectify)
    second_pool = lasagne.layers.MaxPool2DLayer(second_conv, context_pool_rate)
    third_conv = lasagne.layers.Conv2DLayer(second_pool,
                                             context_number_of_convolving_filters,
                                             context_convolving_filter_size, 
                                             stride = 1,
                                             pad = 'same',
                                             nonlinearity = lasagne.nonlinearities.rectify)
    third_pool = lasagne.layers.MaxPool2DLayer(third_conv, context_pool_rate)
    fc = lasagne.layers.DenseLayer(third_pool, 
                                       glimpse_output_size*recurrent_output_size, 
                                       nonlinearity = lasagne.nonlinearities.rectify)
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


