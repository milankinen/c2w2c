import numpy as np
from keras import backend as K
from keras.layers import Recurrent, initializations, activations, InputSpec


class DecoderGRU(Recurrent):
  """
    Recurrent GRU variant specialized for character decoding, following paper:
    http://arxiv.org/abs/1406.1078#
  """
  def __init__(self, output_dim,
               init='glorot_uniform', inner_init='orthogonal',
               activation='tanh', inner_activation='hard_sigmoid', **kwargs):
    self.output_dim       = output_dim
    self.init             = initializations.get(init)
    self.inner_init       = initializations.get(inner_init)
    self.activation       = activations.get(activation)
    self.inner_activation = activations.get(inner_activation)
    super(DecoderGRU, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape[0])]
    self.input_dim = input_shape[0][2]
    self.embedding_dim = input_shape[1][1]

    self.states = [None]

    self.W = self.init((self.input_dim, 3 * self.output_dim),
                       name='{}_W'.format(self.name))
    self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                             name='{}_U'.format(self.name))

    self.C = self.inner_init((self.embedding_dim, 3 * self.output_dim),
                             name='{}_C'.format(self.name))

    self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                   np.zeros(self.output_dim),
                                   np.zeros(self.output_dim))),
                        name='{}_b'.format(self.name))

    self.trainable_weights = [self.W, self.U, self.C, self.b]

  def reset_states(self):
    pass

  def get_initial_states(self, x):
    return super(DecoderGRU, self).get_initial_states(x[0])

  def preprocess_input(self, x):
    return x[0]

  def call(self, x, mask=None):
    return super(DecoderGRU, self).call(x, mask[0])

  def compute_mask(self, input, mask):
    return mask[0]

  def get_output_shape_for(self, input_shape):
    return super(DecoderGRU, self).get_output_shape_for(input_shape[0])

  def step(self, x, states):
    h_tm1 = states[0]  # previous memory
    c_c   = states[1]
    c_z   = states[2]
    c_r   = states[3]

    matrix_x = K.dot(x, self.W) + self.b
    matrix_inner = K.dot(h_tm1, self.U[:, :2 * self.output_dim])

    x_z = matrix_x[:, :self.output_dim]
    x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
    inner_z = matrix_inner[:, :self.output_dim]
    inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

    z = self.inner_activation(x_z + inner_z + c_z)
    r = self.inner_activation(x_r + inner_r + c_r)

    x_h = matrix_x[:, 2 * self.output_dim:]
    inner_h = r * (K.dot(h_tm1, self.U[:, 2 * self.output_dim:]) + c_c)
    hh = self.activation(x_h + inner_h)

    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  def get_constants(self, x):
    matrix_c = K.dot(x[1], self.C)
    c_c = matrix_c[:, 2 * self.output_dim:]
    c_z = matrix_c[:, :self.output_dim]
    c_r = matrix_c[:, self.output_dim: 2 * self.output_dim]
    return [c_c, c_z, c_r]

  def get_config(self):
    config = {'output_dim': self.output_dim,
              'embedding_dim': self.embedding_dim,
              'init': self.init.__name__,
              'inner_init': self.inner_init.__name__,
              'activation': self.activation.__name__,
              'inner_activation': self.inner_activation.__name__}
    base_config = super(DecoderGRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
