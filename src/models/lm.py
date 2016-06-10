from keras.layers import LSTM, Input
from keras.models import Model


def LanguageModel(params, V_C, state_seq=False):
  """
    state_seq  :: True returns all intermediate states, False returns only the state (= word embedding)
                  of the predicted word
  """
  context     = Input(shape=(None, params.d_W), dtype='float32')
  s_W         = LSTM(params.d_W, return_sequences=state_seq, dropout_W=0.2, dropout_U=0.2)(context)

  return Model(input=context, output=s_W)
