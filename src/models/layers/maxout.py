from keras import backend as K
from keras.layers import initializations, InputSpec, Layer


class Maxout(Layer):
  """
    Maxout dense layer tailored for W2C decoder output, following paper:
    http://arxiv.org/abs/1406.1078
  """
  def __init__(self, output_dim, nb_feature=4, init='glorot_uniform', bias=True, input_dim=None, **kwargs):
    self.output_dim = output_dim
    self.nb_feature = nb_feature
    self.init = initializations.get(init)

    self.bias = bias
    self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]

    self.input_dim = input_dim
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(Maxout, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(dtype=K.floatx(),
                                 shape=(None, input_shape[0][1], input_shape[0][2])),
                       InputSpec(dtype=K.floatx(),
                                 shape=(None, input_shape[1][1], input_shape[1][2])),
                       InputSpec(dtype=K.floatx(),
                                 shape=(None, input_shape[2][1], input_shape[2][2]))]

    self.W_h = self.init((self.nb_feature, input_shape[0][2], self.output_dim),
                         name='{}_W_h'.format(self.name))

    self.W_y = self.init((self.nb_feature, input_shape[1][2], self.output_dim),
                         name='{}_W_y'.format(self.name))

    self.W_c = self.init((self.nb_feature, input_shape[2][2], self.output_dim),
                         name='{}_W_c'.format(self.name))

    trainable = [self.W_h, self.W_y, self.W_c]

    if self.bias:
      self.b = K.zeros((self.nb_feature, self.output_dim),
                       name='{}_b'.format(self.name))
      self.trainable_weights = trainable + [self.b]
    else:
      self.trainable_weights = trainable

  def get_output_shape_for(self, input_shape):
    assert input_shape and len(input_shape) == 3
    input_shape = input_shape[0]
    return input_shape[0], input_shape[1], self.output_dim

  def call(self, x, mask=None):
    def flatten(i):
      input_shape = self.input_spec[i].shape
      X = x[i]
      return K.reshape(X, (-1, ) + input_shape[2:])

    # no activation, this layer is only linear.
    output = K.dot(flatten(0), self.W_h) + K.dot(flatten(1), self.W_y) + K.dot(flatten(2), self.W_c)
    if self.bias:
      output += self.b

    output = K.max(output, axis=1)
    output_shape = self.get_output_shape_for([self.input_spec[0].shape] * 3)
    return K.reshape(output, (-1, self.input_spec[0].shape[1]) + output_shape[2:])

  def get_config(self):
    config = {'output_dim': self.output_dim,
              'init': self.init.__name__,
              'nb_feature': self.nb_feature,
              'bias': self.bias,
              'input_dim': self.input_dim}
    base_config = super(Maxout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
