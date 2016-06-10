from keras.layers import LSTM, Embedding, Input, Dense, merge
from keras.models import Model


def C2W(params, V_C):
  indices   = Input(shape=(params.maxlen,), dtype='int32')
  c_E       = Embedding(V_C.size + 1, params.d_C, mask_zero=True)(indices)

  forward   = LSTM(params.d_Wi, go_backwards=False)(c_E)
  backwards = LSTM(params.d_Wi, go_backwards=True)(c_E)

  s_Ef      = Dense(params.d_W)(forward)
  s_Eb      = Dense(params.d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  return Model(input=indices, output=s_E)
