from .orthogonal_networks import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import control as ctrl

class OrthogonalCell(tf.keras.layers.Layer):
    def __init__(self,
        units = 1,
        order = 2,
        input_dims = 2,
        variant = "ct_laguerre",
        disc_method = "zoh",
        dt = 1,
        p = 1,
        a = 0.1,
        theta = 1,
        alpha_initializer = "lecun_uniform",
        hidden_activation = "tanh",
        kernel_initializer = "glorot_uniform",
        recurrent_initializer = "orthogonal",
        bias_initializer = "zeros",
        trainable_C = False,
        verbose = False,
        **kwargs):

        self.units = units
        self.order= order
        self.input_dims = input_dims
        self.variant = variant
        self.disc_method = disc_method
        self.dt = dt
        self.p = p
        self.a = a
        self.theta = theta
        self.hidden_activation = tf.keras.activations.get(hidden_activation)

        self.alpha_initializer = alpha_initializer
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.trainable_C = trainable_C
        self.verbose = verbose

        self.state_size = (self.units, self.input_dims, self.order)
        self.output_size = self.units

        super().__init__(**kwargs)

    def build(self, input_shape):

        # Validation of the variant of the cell
        if self.variant == "ct_laguerre":
            # Computing CT Laguerre Network
            self._A_ct, self._B_ct = laguerre_network_ct(self.p, self.order, input_shape[-1])
            self._C_ct = np.ones(shape = (input_shape[-1], self.order))

            self._ss_ct = ctrl.StateSpace(self._A_ct, self._B_ct, self._C_ct, np.zeros(shape = (input_shape[-1], input_shape[-1])))

            self._ss_dt = self._ss_ct.sample(self.dt, self.disc_method)

            self._A, self._B = self._ss_dt.A, self._ss_dt.B
            self.trainable_C = True

        elif self.variant == "dt_laguerre":

            # Direct computation of DT laguerre network
            self._A, self._B = laguerre_network_dt(self.a, self.order, input_shape[-1])
            self.trainable_C = True

        elif self.variant == "ct_legendre":
            self._A_ct, self._B_ct, self._C_ct, self._D_ct = legendre_pade(self.theta, self.order, input_shape[-1])

            self._ss_ct = ctrl.StateSpace(self._A_ct, self._B_ct, self._C_ct, self._D_ct)

            self._ss_dt = self._ss_ct.sample(self.dt, self.disc_method)

            self._A, self._B = self._ss_dt.A, self._ss_dt.B
            self.trainable_C = True

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

        self.CT = self.add_weight(name = "CT", shape = (self.order, input_shape[-1]), initializer = self.kernel_initializer, trainable = self.trainable_C)
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
                order = self.order,
                input_dims = self.input_dims,
                variant = self.variant,
                disc_method = self.disc_method,
                dt = self.dt,
                p = self.p,
                a = self.a,
                theta = self.theta,
                hidden_activation = self.hidden_activation,
                trainable_C = self.trainable_C,
                verbose = self.verbose
            )
        )

        return config
