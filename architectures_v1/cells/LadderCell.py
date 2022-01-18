from .orthogonal_networks import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import control as ctrl

class LadderCell(tf.keras.layers.Layer):
    def __init__(self,
        units = 1,
        max_delay = 100,
        input_dims = 2,
        hidden_activation = "tanh",
        kernel_initializer = 'glorot_uniform',
        recurrent_initializer = 'orthogonal',
        bias_initializer = 'zeros',
        verbose = False,
        **kwargs):

        self.units = units
        self.max_delay = max_delay
        self.input_dims = input_dims
        self.n_outputs = input_dims
        self.hidden_activation = tf.keras.activations.get(hidden_activation)
        self.verbose = verbose

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # Construction of the cell
        self._A, self._B, self._C, _ = ladder_network(self.input_dims, self.n_outputs, self.max_delay)
        self.order = self._A.shape[0]

        self.state_size = (self.units, self.input_dims, self.order)
        self.output_size = self.units

        super().__init__(**kwargs)

    def build(self, input_shape):

        # State layer weights
        #self.alpha_u = self.add_weight(name = "alpha_u", shape = (input_shape[-1], ), initializer = self.alpha_initializer, trainable = False)
        #self.alpha_m = self.add_weight(name = "alpha_m", shape = (input_shape[-1], ), initializer = self.alpha_initializer, trainable = False)

        # Memory state weights
        self.wfm = self.add_weight(name = "wfm", shape = (input_shape[-1], ), initializer = self.kernel_initializer, trainable = True)

        # Nonlinear output layer weights
        self.WyhT = self.add_weight(name = "WyhT", shape = (self.units, self.units), initializer = self.recurrent_initializer, trainable = True)
        self.WyxT = self.add_weight(name = "WyxT", shape = (self.order, self.units), initializer = self.kernel_initializer, trainable = True)
        self.WyfT = self.add_weight(name = "WyfT", shape = (input_shape[-1], self.units), initializer = self.kernel_initializer, trainable = True)

        # Nonlinear output layer biases
        self.by = self.add_weight(shape = (self.units, ), initializer = self.bias_initializer, trainable = True)

        # A, B and C matrices
        self.AT = self.add_weight(name = "AT", shape = (self.order, self.order), initializer = Constant(self._A.T), trainable = False)
        if self.verbose: print(f"AT : {self.AT.shape}")

        self.BT = self.add_weight(name = "BT", shape = (input_shape[-1], self.order), initializer = Constant(self._B.T), trainable = False)
        if self.verbose: print(f"BT : {self.BT.shape}")

        self.CT = self.add_weight(name = "CT", shape = (self.order, input_shape[-1]), initializer = Constant(self._C.T), trainable = False)
        if self.verbose: print(f"CT : {self.CT.shape}")

        super().build(input_shape)

    def call(self, inputs, states):

        inp = inputs
        if self.verbose: print(f"inp: {inp.shape}")

        h, m, x = states
        if self.verbose: print(f"h: {h.shape}")
        if self.verbose: print(f"x: {x.shape}")
        if self.verbose: print(f"m: {m.shape}")

        # Input to the state equation
        #u = self.alpha_u * inp + self.alpha_m * m
        u = inp + m
        if self.verbose: print(f"u: {u.shape}")

        # State equation
        x = tf.keras.backend.dot(x, self.AT) + tf.keras.backend.dot(u, self.BT)
        if self.verbose: print(f"x (after update): {x.shape}")

        # Memory equation
        m = tf.keras.backend.dot(x, self.CT)
        if self.verbose: print(f"m (after update): {m.shape}")

        f = self.wfm * m
        if self.verbose: print(f"f (after computation): {f.shape}")

        # Output equation
        h = self.hidden_activation(tf.keras.backend.dot(h, self.WyhT) + tf.keras.backend.dot(x, self.WyxT) + tf.keras.backend.dot(f, self.WyfT) + self.by)

        return h, [h, m, x]

    def get_config(self):
        """
        Overrides the tensorflow get_config function.
        """

        config = super().get_config()
        config.update(
            dict(
                units = self.units,
                max_delay = self.max_delay,
                input_dims = self.input_dims,
                hidden_activation = self.hidden_activation,
                verbose = self.verbose
            )
        )

        return config
