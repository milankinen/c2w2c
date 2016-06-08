from keras.layers import LSTM, TimeDistributed, Input, Dense, RepeatVector
from keras.models import Model


def W2C(V_C, V_W, d_W, d_C):
  """
    V_C        :: character vocabulary
    V_W        :: word vocabulary
    d_W        :: word embedding dimension ("word properties")
    d_C        :: character embedding dimension ("character properties")
  """

  w_E = Input(shape=(d_W,), dtype='float32')

  # V_W.dim[1] = max word length
  # V_C.dim[0] = number of characters in vocabulary
  c_E = LSTM(d_C, return_sequences=True)(RepeatVector(V_W.dim[1])(w_E))
  c_I = TimeDistributed(Dense(V_C.dim[0], activation='softmax'))(c_E)

  return Model(input=w_E, output=c_I)
