from keras.layers import LSTM, Embedding, Input, Dense, Activation, merge
from keras.models import Model


def C2W(V_C, V_W, d_Wi, d_W, d_C, activation=None):
  """
    V_C        :: character vocabulary
    V_W        :: word vocabulary
    d_Wi       :: intermediate word state sequence dimension
    d_W        :: word embedding dimension ("word properties")
    d_C        :: character embedding dimension ("character properties")
    activation :: activation to be used if model is trained separately
  """

  # V_W.dim[1] = max word length
  # V_C.dim[0] = number of characters in vocabulary
  indices   = Input(shape=(V_W.dim[1],), dtype='int32')
  c_E       = Embedding(V_C.dim[0], d_C)(indices)

  forward   = LSTM(d_Wi, go_backwards=False)(c_E)
  backwards = LSTM(d_Wi, go_backwards=True)(c_E)

  s_Ef      = Dense(d_W)(forward)
  s_Eb      = Dense(d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  if activation is not None:
    s_E = Activation(activation)(s_E)

  return Model(input=indices, output=s_E)
