import tensorflow as tf
import numpy as np

# NBRC - TensorFlow 1
class NeuromodulatedBistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        
        self.output_size = output_dim
        self.state_size = output_dim
        
        super().__init__(**kwargs)

    def build(self, input_shape):
        
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.output_size, self.output_size), dtype=tf.float32,
                                      initializer='orthogonal')
        self.memoryr = self.add_weight(name="mr", shape=(self.output_size, self.output_size), dtype=tf.float32,
                                      initializer='orthogonal')

        self.br = self.add_weight(name="br", shape=(self.output_size,), dtype = tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(self.output_size,), dtype = tf.float32, initializer='zeros')

        super().build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + tf.matmul(prev_out, self.memoryz) + self.bz)
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + tf.matmul(prev_out, self.memoryr) + self.br)+1
        h = tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out
        return output, [output]
    
    def get_config(self):
        """
        Overrides the tensorflow get_config function.
        """

        config = super().get_config()
        config.update(
            dict(
                output_dim = self.output_size
            )
        )

        return config

# BRC - TensorFlow 1
class BistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        
        self.output_size = output_dim
        self.state_size = output_dim
        
        super().__init__(output_dim, **kwargs)

    def build(self, input_shape):
        
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_size), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.output_size,), dtype=tf.float32,initializer=tf.keras.initializers.constant(1.0))
        self.memoryr = self.add_weight(name="mr", shape=(self.output_size,), dtype=tf.float32,initializer=tf.keras.initializers.constant(1.0))

        self.br = self.add_weight(name="br", shape=(self.output_size,), dtype = tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(self.output_size,), dtype = tf.float32, initializer='zeros')

        super().build(input_shape)

    def call(self, input, states):
        
        inp = input
        prev_out = states[0]
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + prev_out * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + prev_out * self.memoryz + self.bz)
        output = z * prev_out + (1.0 - z) * tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        return output, [output]
    
    def get_config(self):
        """
        Overrides the tensorflow get_config function.
        """

        config = super().get_config()
        config.update(
            dict(
                output_dim = self.output_size
            )
        )

        return config