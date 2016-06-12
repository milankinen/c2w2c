from keras.layers import LSTM, TimeDistributed, Input, Dense, RepeatVector, merge
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
    e_input = Input(shape=(params.d_W,), dtype='float32')
  if p_input is None:
    p_input = Input(shape=(params.maxlen, V_C.size), dtype='int8', name='predicted_word')

  c_E     = TimeDistributed(Projection(params.d_C))(p_input)
  w_E     = RepeatVector(params.maxlen)(e_input)
  w_EC    = merge(inputs=[w_E, c_E], mode='concat')
  c_E     = LSTM(params.d_D,
                 return_sequences=True,
                 dropout_W=0.2,
                 dropout_U=0.2)(w_EC)
  c_I     = TimeDistributed(Dense(V_C.size, activation='softmax'))(c_E)

  return Model(input=[e_input, p_input], output=c_I)
