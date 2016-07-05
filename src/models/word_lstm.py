from keras.layers import Input, Dropout, Activation, Dense, TimeDistributed
from keras.models import Model

from sub_models import LanguageModel
from layers import ProjectionOverTime


def WordLSTM(batch_size, n_ctx, params, V_W):
  """
    Returns tuple (word_lstm model, sub-models, inputs)
  """
  # params
  d_W       = params.d_W
  d_L       = params.d_L

  # inputs
  ctx       = Input(batch_shape=(batch_size, n_ctx, V_W.size), name='ctx')
  ctx_mask  = Input(batch_shape=(batch_size, n_ctx), name='ctx_mask', dtype='int8')

  # sub-models
  lm        = LanguageModel(batch_size, n_ctx, d_W, d_L)

  # the actual word_lstm model
  W_e       = Dropout(.5)(TimeDistributed(Dense(d_W))(ctx))
  C         = Dropout(.5)(lm([W_e, ctx_mask]))
  C_i       = ProjectionOverTime(150)(C)
  Y_t       = Activation('softmax')(TimeDistributed(Dense(V_W.size))(C_i))
  word_lstm = Model(input=[ctx, ctx_mask], output=Y_t)

  return word_lstm

