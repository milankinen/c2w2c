from keras.layers import LSTM, Input, Dense, TimeDistributed, Masking, merge
from keras.models import Model
from layers import Projection


def C2W(params, V_C):
  one_hots  = Input(shape=(params.maxlen, V_C.size), dtype='int8')
  c_E       = TimeDistributed(Projection(params.d_C))(one_hots)
  # we want to preserve the state in case of padding so that the state
  # sequence s_Ef and s_Eb last values remain correct
  c_E_mask  = Masking(mask_value=0.)(c_E)

  forward   = LSTM(params.d_Wi, go_backwards=False, consume_less='gpu')(c_E_mask)
  backwards = LSTM(params.d_Wi, go_backwards=True, consume_less='gpu')(c_E_mask)

  s_Ef      = Dense(params.d_W)(forward)
  s_Eb      = Dense(params.d_W)(backwards)
  s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

  return Model(input=one_hots, output=s_E, name='W2C')
