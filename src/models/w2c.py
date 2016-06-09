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
  c_E = LSTM(1024, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(RepeatVector(V_W.maxlen)(w_E))
  c_I = TimeDistributed(Dense(V_C.size, activation='softmax'))(c_E)

  return Model(input=w_E, output=c_I)
