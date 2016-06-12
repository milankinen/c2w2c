from keras import backend as K
from keras import initializations, activations
from keras.engine.topology import Layer, InputSpec


class Projection(Layer):
  """
    Simple (character) projection layer without bias so that
    it preserves input maskability (zero one-hots are projected
    into zero vectors)
  """
  def __init__(self, output_dim, weights=None, **kwargs):
    self.output_dim       = output_dim
    self.init             = initializations.get('glorot_uniform')
    self.activation       = activations.get('linear')
    self.initial_weights  = weights
    super(Projection, self).__init__(**kwargs)

  def build(self, input_shape):
    input_dim = input_shape[1]
    self.input_spec = [InputSpec(dtype=K.floatx(),
                                 shape=(None, input_dim))]
    self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
    self.trainable_weights = [self.W]
    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights

  def call(self, x, mask=None):
    return K.dot(x, self.W)

  def get_output_shape_for(self, input_shape):
    return input_shape[0], self.output_dim
