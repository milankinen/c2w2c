from keras.layers import LSTM, TimeDistributed, Input, Dense, RepeatVector, merge, Activation
from keras.models import Model
from layers import Projection


def W2C(params, V_C, e_input=None, p_input=None):
  """
  params    :: model params
  V_C       :: character vocabulary
  P_C       :: projection layer from character lookup table (from C2W)
  p_input   :: input for predicted characters (one-hots)
  """
  if e_input is None:
    e_input = Input(shape=(params.d_W,), dtype='floatX')
  if p_input is None:
    p_input = Input(shape=(params.maxlen, V_C.size), dtype='int8', name='predicted_word')

  w_E     = RepeatVector(params.maxlen)(e_input)
  w_EC    = merge(inputs=[w_E, p_input], mode='concat')
  c_E     = LSTM(params.d_D,
                 return_sequences=True,
                 dropout_U=0.1,
                 dropout_W=0.1,
                 consume_less='gpu')(w_EC)
  c_I     = TimeDistributed(Dense(V_C.size, activation='softmax'))(c_E)

  return Model(input=[e_input, p_input], output=c_I, name='W2C')
