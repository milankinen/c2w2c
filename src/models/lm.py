from keras.layers import LSTM, Input, TimeDistributed, Dropout
from keras.models import Model
from layers import Projection


def LanguageModel(params, V_C, state_seq=False):
  """
    state_seq  :: True returns all intermediate states, False returns only the state (= word embedding)
                  of the predicted word
  """
  context     = Input(shape=(None, params.d_W), dtype='floatX')
  s_Wi        = LSTM(params.d_L,
                     return_sequences=state_seq,
                     consume_less='gpu',
                     dropout_W=0.25,
                     dropout_U=0.25)(context)

  if state_seq is True:
    # for testing
    s_Wnp1    = TimeDistributed((Projection(params.d_W)))(s_Wi)
  else:
    # for training
    s_Wnp1    = Dropout(0.25)(Projection(params.d_W)(s_Wi))

  return Model(input=context, output=s_Wnp1, name='LM')
