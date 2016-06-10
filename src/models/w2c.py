from keras.layers import LSTM, TimeDistributed, Input, Dense, RepeatVector
from keras.models import Model


def W2C(params, V_C):
  w_E = Input(shape=(params.d_W,), dtype='float32')
  c_E = LSTM(params.d_D, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(RepeatVector(params.maxlen)(w_E))
  c_I = TimeDistributed(Dense(V_C.size, activation='softmax'))(c_E)

  return Model(input=w_E, output=c_I)
