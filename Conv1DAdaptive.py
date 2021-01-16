
import os
import six


from keras_utils_tf2 import ClipCallback, PrintLayerVariableStats

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, regularizers, constraints
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec
# from keras.initializers import glorot_uniform as w_ini
from tensorflow.keras import backend as K

# pylint: enable=unused-import


from tensorflow.python.framework import tensor_shape
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops, nn_ops
from tensorflow.python.ops import nn


def idx_init(shape, dtype='float32'):
    idxs = np.zeros((shape[0], shape[1]), dtype)
    c = 0
    # assumes square filters

    wid = np.int(shape[0])

    f = np.float32
    for x in np.arange(wid):  # / (self.incoming_width * 1.0):
        idxs[c,] = np.array([x / f(wid - 1)], dtype)
        c += 1

    return idxs


class Conv1DAdaptive(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 rank=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 init_sigma=0.1,
                 groups=1,
                 norm=2,
                 activation=None,
                 trainSigmas=True,
                 trainWeights=True,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 sigma_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 conv_op=None,
                 **kwargs):
        super(Conv1DAdaptive, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.initsigma = None
        self.rank = rank

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.initsigma = init_sigma
        self.norm = norm
        self.sigma_regularizer = regularizers.get(sigma_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.trainSigmas = trainSigmas
        self.trainWeights = trainWeights
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.input_channels = input_dim
        input_shape = tensor_shape.TensorShape(input_shape)

        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.idxs = idx_init(shape=[self.kernel_size[0],1])
        self.mu = 0.5

        self.W = self.add_weight(
            name='weights',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        s_init = self.sigma_initializer  # ((self.filters,),dtype='float32')

        self.Sigma = self.add_weight(shape=(self.filters,),
                                     name='Sigma',
                                     initializer=s_init,
                                     trainable=self.trainSigmas,
                                     constraint=None,
                                     regularizer=self.sigma_regularizer,
                                     dtype='float32')
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1DAdaptive':
            tf_op_name = 'conv1dAdaptive'  # Backwards compat.

    @tf.function
    def calcU(self):
        up = (self.idxs - self.mu) ** 2
        dwn = 2 * (self.Sigma ** 2) + 1e-8  # 1e-8 is to prevent div by zero

        result = tf.exp(-up / dwn)
        # now the maximum is 1

        # we do not care about input channels, later it will be broadcasted to input size.
        masks = tf.reshape(result, (self.kernel_size[0],
                                    1, self.filters))

        masks = tf.repeat(masks, repeats=self.input_channels, axis=1)

        # Normalize to 1
        if self.norm == 1:
            masks /= tf.sqrt(tf.reduce_sum(K.square(masks), axis=(0, 1), keepdims=True))
        elif self.norm == 2:
            # make sum(x**2)==1
            masks /= tf.sqrt(tf.reduce_sum(masks ** 2, axis=(0, 1), keepdims=True))
            # tf.print("Sum:     ",tf.reduce_sum(masks**2,axis=[0,1])[0:10])
            # input('zasd')
            masks *= tf.sqrt(tf.constant(self.input_channels * self.kernel_size[0], dtype=tf.float32))
            # tf.print("Sum:     ",tf.reduce_sum(masks**2,axis=[0,1])[0:10])
            # tf.print("Max:     ",tf.reduce_max(masks,axis=[0,1])[0:10])

        return masks

    def sigma_initializer(self, shape, dtype='float32'):
        initsigma = self.initsigma

        print("Initializing sigma", type(initsigma), initsigma, type(dtype))

        if isinstance(initsigma, float):  # initialize it with the given scalar
            sigma = initsigma * np.ones(shape[0])
        elif (isinstance(initsigma, tuple) or isinstance(initsigma, list)) and len(initsigma) == 2:  # linspace in range
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0])
        elif isinstance(initsigma, np.ndarray) and initsigma.shape[1] == 2 and shape[
            0] != 2:  # set the values directly from array
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0])
        elif isinstance(initsigma, np.ndarray):  # set the values directly from array
            sigma = np.convert_to_tensor(initsigma)
        else:
            print("Default initial sigma value 0.1 will be used")
            sigma = np.float32(0.1) * np.ones(shape[0])

        # print("Scale initializer:",sigma)
        return sigma

    def call(self, inputs):
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
        self.U = self.calcU()
        kernel = self.W * self.U
        outputs = K.conv1d(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            if self.data_format == 'channels_first':
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        else:
            return tensor_shape.TensorShape(
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
        return False


    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'rank':
                self.rank,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'init_sigma':
                self.initsigma,
            'groups':
                self.groups,
            'norm':
                self.norm,
            'activation':
                activations.serialize(self.activation),
            'trainSigmas':
                self.trainSigmas,
            'trainWeights':
                self.trainWeights,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'sigma_regularizer':
                regularizers.serialize(self.sigma_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1DAdaptive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


if __name__ == '__main__':

    # CNN for the IMDB problem
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing import sequence

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    # pad dataset to a maximum review length in words
    max_words = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    print(X_train.shape)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    MIN_SIG = 1.0/7
    MAX_SIG = 7*1.0

    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]



    # create the model
    def experiment_run(acnn):
        model = Sequential()
        callbacks = []
        model.add(Embedding(top_words, 32, input_length=max_words))
        if acnn:
            model.add(Conv1DAdaptive(32, 7, padding='same', activation='relu',name="acnn-1",init_sigma=[MIN_SIG, MAX_SIG]))
            ccp1 = ClipCallback('Sigma', [MIN_SIG, MAX_SIG])
            pr_1 = PrintLayerVariableStats("acnn-1", "Weights:0", stat_func_list, stat_func_name)
            pr_2 = PrintLayerVariableStats("acnn-1", "Sigma:0", stat_func_list, stat_func_name)
            callbacks += [ccp1,pr_1,pr_2]
        else:
            model.add(Conv1D(32, 7, padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipvalue=1.0)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2,callbacks=callbacks)
        scores = model.evaluate(X_test, y_test, verbose=0)
        # Final evaluation of the model
        if acnn:
            print("Accuracy of Adaptive 1D Conv: %.2f%%" % (scores[1] * 100))
        else:
            print("Accuracy of Ordinary 1D Conv: %.2f%%" % (scores[1] * 100))

    experiment_run(True)
    experiment_run(False)

