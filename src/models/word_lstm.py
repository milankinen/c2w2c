from keras import backend as K
from keras.layers import Input, Dropout, Activation, Dense, Layer
from keras.models import Model

from sub_models import LanguageModel


def WordLSTM(batch_size, d_W, d_L, V_W):
  class WordMask(Layer):
    def __init__(self, **kwargs):
      super(WordMask, self).__init__(**kwargs)

    def call(self, x, mask=None):
      assert mask is None
      return K.cast(K.any(x, axis=-1), K.floatx())

    def get_output_shape_for(self, input_shape):
      return input_shape

  # inputs
  x = Input(batch_shape=(batch_size, V_W.size), name='context')

  # sub-models
  input     = Dense(d_W)
  lm        = LanguageModel(batch_size, d_W, d_L)
  output    = Dense(V_W.size)

  # the actual word_lstm model
  ctx       = Dropout(.5)(input(x))
  ctx_mask  = WordMask()(x)
  c         = Dropout(.5)(lm([ctx, ctx_mask]))
  y_logit   = output(Dense(150)(c))
  y         = Activation('softmax')(y_logit)
  word_lstm = Model(input=x, output=y)

  return word_lstm

