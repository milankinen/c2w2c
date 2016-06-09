from keras.layers import LSTM, Embedding, Input, Dense, merge
from keras.models import Model


def C2W(V_C, V_W, d_Wi, d_W, d_C):
  """
    V_C        :: character vocabulary
    V_W        :: word vocabulary
    d_Wi       :: intermediate word state sequence dimension
    d_W        :: word embedding dimension ("word properties")
    d_C        :: character embedding dimension ("character properties")
  """

  indices   = Input(shape=(V_W.maxlen,), dtype='int32')
  c_E       = Embedding(V_C.size, d_C)(indices)

  forward   = LSTM(d_Wi, go_backwards=False)(c_E)
  backwards = LSTM(d_Wi, go_backwards=True)(c_E)

  s_Ef      = Dense(d_W)(forward)
  s_Eb      = Dense(d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  return Model(input=indices, output=s_E)
