from keras.layers import LSTM, Input
from keras.models import Model


def LanguageModel(d_W, state_seq=False):
  """
    d_W        :: word embedding dimension ("word properties")
    state_seq  :: True returns all intermediate states, False returns only the state (= word embedding)
                  of the predicted word
  """
  context     = Input(shape=(None, d_W), dtype='float32')
  s_W         = LSTM(d_W, return_sequences=state_seq, dropout_W=0.2, dropout_U=0.2)(context)

  return Model(input=context, output=s_W)
