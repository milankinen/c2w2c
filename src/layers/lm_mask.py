from keras import backend as K
from keras.layers import Masking


class LMMask(Masking):
  """
    Special Language Model masking layer that takes the actual word embedding of
    w_n as a first input AND mask bit as a second input, masks the word embedding
    and returns output mask based on mask bits.
  """

  def __init__(self, mask_value=0, **kwargs):
    super(LMMask, self).__init__(**kwargs)
    self.mask_value = mask_value

  def get_output_shape_for(self, input_shape):
    return input_shape[0]

  def compute_mask(self, x, input_mask=None):
    m = K.reshape(x[1], (-1, 1, 1))
    return K.cast(m, K.floatx())

  def call(self, x, mask=None):
    m = K.reshape(x[1], (-1, 1, 1))
    mf = K.cast(K.repeat_elements(m, x[0].shape[-1], axis=2), K.floatx())
    return x[0] * mf
