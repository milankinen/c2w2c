from keras.layers import Input, Dropout, Activation
from keras.models import Model

from models import C2W, LanguageModel, W2W


def C2W2W(n_batch, params, V_C, V_W, c2w_trainable=True, lm_trainable=True, w2c_trainable=True):
  """
    Returns tuple (c2w2w model, sub-models, inputs)
  """
  # params
  maxlen    = params.maxlen
  d_C       = params.d_C
  d_Wi      = params.d_Wi
  d_W       = params.d_W
  d_L       = params.d_L

  # inputs
  w_nc      = Input(batch_shape=(n_batch, maxlen, V_C.size), name='w_nc', dtype='int8')
  w_nmask   = Input(batch_shape=(n_batch, 1), name='w_nmask', dtype='int8')

  # sub-models
  c2w       = C2W(maxlen, d_C, d_W, d_Wi, V_C, trainable=c2w_trainable)
  lm        = LanguageModel(n_batch, d_W, d_L, trainable=lm_trainable)
  w2w       = W2W(n_batch, d_L, V_W, trainable=w2c_trainable)

  # the actual c2w2w model
  w_nE      = Dropout(.5)(c2w(w_nc))
  w_np1E    = Dropout(.5)(lm([w_nE, w_nmask]))
  w_np1     = Activation('softmax')(w2w(w_np1E))
  c2w2w     = Model(input=[w_nc, w_nmask], output=w_np1)

  return c2w2w

