from keras import backend as K
from keras import initializations, activations
from keras.engine.topology import Layer, InputSpec
from keras.layers import TimeDistributed


class Projection(Layer):
  """
    Simple (character) projection layer without bias so that
    it preserves input maskability (zero one-hots are projected
    into zero vectors).

    Basically same as Dense without bias but supports zero masking
  """
  def __init__(self, output_dim, weights=None, activation='linear', return_mask=True, **kwargs):
    self.supports_masking = True
    self.output_dim       = output_dim
    self.init             = initializations.get('glorot_uniform')
    self.activation       = activations.get(activation)
    self.initial_weights  = weights
    self.return_mask      = return_mask
    super(Projection, self).__init__(**kwargs)

  def compute_mask(self, input, input_mask=None):
    if self.return_mask:
      return super(Projection, self).compute_mask(input, input_mask)
    else:
      return None

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
    return self.activation(K.dot(x, self.W))

  def get_output_shape_for(self, input_shape):
    return input_shape[0], self.output_dim


class ProjectionOverTime(TimeDistributed):
  """
    Efficient time distributed projection layer that always uses reshaping
    for the time distribution effect (= faster)
  """
  def __init__(self, output_dim, weights=None, activation='linear', return_mask=True, **kwargs):
    layer = Projection(output_dim, weights, activation, return_mask=return_mask)
    super(ProjectionOverTime, self).__init__(layer, **kwargs)

  def call(self, X, mask=None):
    input_shape   = self.input_spec[0].shape
    output_shape  = self.get_output_shape_for(input_shape)
    input_length  = input_shape[1]
    if not input_length:
      input_length = K.shape(X)[1]

    # (nb_samples, timesteps, ...) => (nb_samples * timesteps, ...)
    X = K.reshape(X, (-1, ) + input_shape[2:])
    y = self.layer.call(X, mask)

    # (nb_samples * timesteps, ...) => (nb_samples, timesteps, ...)
    return K.reshape(y, (-1, input_length) + output_shape[2:])

