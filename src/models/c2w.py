import keras.backend as K

from keras.layers import LSTM, Embedding, Input, Dense, merge
from keras.models import Model


class CharEmbedding(Embedding):
  def __init__(self, input_dim, output_dim, **kwargs):
    super(CharEmbedding, self).__init__(input_dim, output_dim, mask_zero=True, **kwargs)

  def compute_mask(self, x, mask=None):
    return K.equal(x, -1)


def C2W(V_C, V_W, d_Wi, d_W, d_C):
  """
    V_C        :: character vocabulary
    V_W        :: word vocabulary
    d_Wi       :: intermediate word state sequence dimension
    d_W        :: word embedding dimension ("word properties")
    d_C        :: character embedding dimension ("character properties")
  """

  indices   = Input(shape=(V_W.maxlen,), dtype='int32')
  c_E       = CharEmbedding(V_C.size, d_C)(indices)

  forward   = LSTM(d_Wi, go_backwards=False)(c_E)
  backwards = LSTM(d_Wi, go_backwards=True)(c_E)

  s_Ef      = Dense(d_W)(forward)
  s_Eb      = Dense(d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  return Model(input=indices, output=s_E)
