from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay

# Importing the Laguerre Delay function
from LaguerreDelay import *

import numpy as np

from keras import backend as K
from keras import activations, initializers
from keras.initializers import Constant, Initializer
from keras.layers import Layer

class Legendre(Initializer):

    def __call__(self, shape, dtype=None):
        from scipy.special import legendre
        if len(shape) != 2:
            raise ValueError("Legendre initializer assumes shape is 2D; "
                             "but shape=%s" % (shape,))
        # TODO: geometric spacing might be useful too!
        return np.asarray([legendre(i)(np.linspace(-1, 1, shape[1]))
                           for i in range(shape[0])])

class LMUCell(Layer):

    def __init__(self,
                 units,
                 order,
                 theta,  # relative to dt=1
                 method='zoh',
                 realizer=Identity(),    # TODO: Deprecate?
                 factory=LegendreDelay,  # TODO: Deprecate?
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer='lecun_uniform',
                 hidden_encoders_initializer='lecun_uniform',
                 memory_encoders_initializer=Constant(0),  # 'lecun_uniform',
                 input_kernel_initializer='glorot_normal',
                 hidden_kernel_initializer='glorot_normal',
                 memory_kernel_initializer='glorot_normal',
                 hidden_activation='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.order = order
        self.theta = theta
        self.method = method
        self.realizer = realizer
        self.factory = factory
        self.trainable_input_encoders = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel = trainable_input_kernel
        self.trainable_hidden_kernel = trainable_hidden_kernel
        self.trainable_memory_kernel = trainable_memory_kernel
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        self.input_encoders_initializer = initializers.get(
            input_encoders_initializer)
        self.hidden_encoders_initializer = initializers.get(
            hidden_encoders_initializer)
        self.memory_encoders_initializer = initializers.get(
            memory_encoders_initializer)
        self.input_kernel_initializer = initializers.get(
            input_kernel_initializer)
        self.hidden_kernel_initializer = initializers.get(
            hidden_kernel_initializer)
        self.memory_kernel_initializer = initializers.get(
            memory_kernel_initializer)

        self.hidden_activation = activations.get(hidden_activation)

        self._realizer_result = realizer(
            factory(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1., method=method)
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        # assert self._C.shape == (1, self.order)
        # C_full = np.zeros((self.units, self.order, self.units))
        # for i in range(self.units):
        #     C_full[i, :, i] = self._C[0]
        # decoder_initializer = Constant(
        #     C_full.reshape(self.units*self.order, self.units))

        # TODO: would it be better to absorb B into the encoders and then
        # initialize it appropriately? trainable encoders+B essentially
        # does this in a low-rank way

        # if the realizer is CCF then we get the following two constraints
        # that could be useful for efficiency
        # assert np.allclose(self._ss.B[1:], 0)  # CCF
        # assert np.allclose(self._ss.B[0], self.order**2)

        self.state_size = (self.units, self.order)
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # TODO: add regularizers

        self.input_encoders = self.add_weight(
            name='input_encoders',
            shape=(input_dim, 1),
            initializer=self.input_encoders_initializer,
            trainable=self.trainable_input_encoders)

        self.hidden_encoders = self.add_weight(
            name='hidden_encoders',
            shape=(self.units, 1),
            initializer=self.hidden_encoders_initializer,
            trainable=self.trainable_hidden_encoders)

        self.memory_encoders = self.add_weight(
            name='memory_encoders',
            shape=(self.order, 1),
            initializer=self.memory_encoders_initializer,
            trainable=self.trainable_memory_encoders)

        self.input_kernel = self.add_weight(
            name='input_kernel',
            shape=(input_dim, self.units),
            initializer=self.input_kernel_initializer,
            trainable=self.trainable_input_kernel)

        self.hidden_kernel = self.add_weight(
            name='hidden_kernel',
            shape=(self.units, self.units),
            initializer=self.hidden_kernel_initializer,
            trainable=self.trainable_hidden_kernel)

        self.memory_kernel = self.add_weight(
            name='memory_kernel',
            shape=(self.order, self.units),
            initializer=self.memory_kernel_initializer,
            trainable=self.trainable_memory_kernel)

        self.AT = self.add_weight(
            name='AT',
            shape=(self.order, self.order),
            initializer=Constant(self._A.T),  # note: transposed
            trainable=self.trainable_A)

        self.BT = self.add_weight(
            name='BT',
            shape=(1, self.order),  # system is SISO
            initializer=Constant(self._B.T),  # note: transposed
            trainable=self.trainable_B)

        self.built = True

    def call(self, inputs, states):
        h, m = states

        u = (K.dot(inputs, self.input_encoders) +
             K.dot(h, self.hidden_encoders) +
             K.dot(m, self.memory_encoders))

        m = m + K.dot(m, self.AT) + K.dot(u, self.BT)

        h = self.hidden_activation(
             K.dot(inputs, self.input_kernel) +
             K.dot(h, self.hidden_kernel) +
             K.dot(m, self.memory_kernel))

        return h, [h, m]


class InputScaled(Initializer):
    """Divides a constant value by the incoming dimensionality."""

    def __init__(self, value=0):
        super(InputScaled, self).__init__()
        self.value = value

    def __call__(self, shape, dtype=None):
        return K.constant(self.value / shape[0], shape=shape, dtype=dtype)
    
class LaguerreCell(Layer):

    def __init__(self,
                 units,
                 order,
                 a,
                 theta,  # relative to dt=1
                 method='zoh', # this is not required since the Laguerre polynomials are passed directly in DT
                 realizer=Identity(),    # TODO: Deprecate?
                 factory=LaguerreDelay, # change to conform the Laguerre cell
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer='lecun_uniform',
                 hidden_encoders_initializer='lecun_uniform',
                 memory_encoders_initializer=Constant(0),  # 'lecun_uniform',
                 input_kernel_initializer='glorot_normal',
                 hidden_kernel_initializer='glorot_normal',
                 memory_kernel_initializer='glorot_normal',
                 hidden_activation='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.order = order
        self.a = a
        self.theta = theta
        self.method = method
        self.realizer = realizer
        self.factory = factory
        self.trainable_input_encoders = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel = trainable_input_kernel
        self.trainable_hidden_kernel = trainable_hidden_kernel
        self.trainable_memory_kernel = trainable_memory_kernel
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        self.input_encoders_initializer = initializers.get(
            input_encoders_initializer)
        self.hidden_encoders_initializer = initializers.get(
            hidden_encoders_initializer)
        self.memory_encoders_initializer = initializers.get(
            memory_encoders_initializer)
        self.input_kernel_initializer = initializers.get(
            input_kernel_initializer)
        self.hidden_kernel_initializer = initializers.get(
            hidden_kernel_initializer)
        self.memory_kernel_initializer = initializers.get(
            memory_kernel_initializer)

        self.hidden_activation = activations.get(hidden_activation)

        self._realizer_result = realizer(
            factory(theta=theta, order=self.order, a = self.a))
        self._ss = self._realizer_result.realization
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        # assert self._C.shape == (1, self.order)
        # C_full = np.zeros((self.units, self.order, self.units))
        # for i in range(self.units):
        #     C_full[i, :, i] = self._C[0]
        # decoder_initializer = Constant(
        #     C_full.reshape(self.units*self.order, self.units))

        # TODO: would it be better to absorb B into the encoders and then
        # initialize it appropriately? trainable encoders+B essentially
        # does this in a low-rank way

        # if the realizer is CCF then we get the following two constraints
        # that could be useful for efficiency
        # assert np.allclose(self._ss.B[1:], 0)  # CCF
        # assert np.allclose(self._ss.B[0], self.order**2)

        self.state_size = (self.units, self.order)
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # TODO: add regularizers

        self.input_encoders = self.add_weight(
            name='input_encoders',
            shape=(input_dim, 1),
            initializer=self.input_encoders_initializer,
            trainable=self.trainable_input_encoders)

        self.hidden_encoders = self.add_weight(
            name='hidden_encoders',
            shape=(self.units, 1),
            initializer=self.hidden_encoders_initializer,
            trainable=self.trainable_hidden_encoders)

        self.memory_encoders = self.add_weight(
            name='memory_encoders',
            shape=(self.order, 1),
            initializer=self.memory_encoders_initializer,
            trainable=self.trainable_memory_encoders)

        self.input_kernel = self.add_weight(
            name='input_kernel',
            shape=(input_dim, self.units),
            initializer=self.input_kernel_initializer,
            trainable=self.trainable_input_kernel)

        self.hidden_kernel = self.add_weight(
            name='hidden_kernel',
            shape=(self.units, self.units),
            initializer=self.hidden_kernel_initializer,
            trainable=self.trainable_hidden_kernel)

        self.memory_kernel = self.add_weight(
            name='memory_kernel',
            shape=(self.order, self.units),
            initializer=self.memory_kernel_initializer,
            trainable=self.trainable_memory_kernel)

        self.AT = self.add_weight(
            name='AT',
            shape=(self.order, self.order),
            initializer=Constant(self._A.T),  # note: transposed
            trainable=self.trainable_A)

        self.BT = self.add_weight(
            name='BT',
            shape=(1, self.order),  # system is SISO
            initializer=Constant(self._B.T),  # note: transposed
            trainable=self.trainable_B)

        self.built = True

    def call(self, inputs, states):
        h, m = states

        u = (K.dot(inputs, self.input_encoders) +
             K.dot(h, self.hidden_encoders) +
             K.dot(m, self.memory_encoders))

        m = m + K.dot(m, self.AT) + K.dot(u, self.BT)

        h = self.hidden_activation(
             K.dot(inputs, self.input_kernel) +
             K.dot(h, self.hidden_kernel) +
             K.dot(m, self.memory_kernel))

        return h, [h, m]