import keras.backend as K

from keras.layers import LSTM, Embedding, Input, Dense, merge
from keras.models import Model


class CharEmbedding(Embedding):
  def __init__(self, input_dim, output_dim, **kwargs):
    super(CharEmbedding, self).__init__(input_dim, output_dim, mask_zero=True, **kwargs)

  def compute_mask(self, x, mask=None):
    return K.not_equal(x, -1)


def C2W(params, V_C, V_W):
  """
    V_C        :: character vocabulary
    V_W        :: word vocabulary
    d_Wi       :: intermediate word state sequence dimension
    d_W        :: word embedding dimension ("word properties")
    d_C        :: character embedding dimension ("character properties")
  """

  indices   = Input(shape=(V_W.maxlen,), dtype='int32')
  c_E       = Embedding(V_C.size + 1, params.d_C, mask_zero=True)(indices)

  forward   = LSTM(params.d_Wi, go_backwards=False)(c_E)
  backwards = LSTM(params.d_Wi, go_backwards=True)(c_E)

  s_Ef      = Dense(params.d_W)(forward)
  s_Eb      = Dense(params.d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  return Model(input=indices, output=s_E)
