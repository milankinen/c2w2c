from keras.layers import Input, Dropout, Activation
from keras.models import Model

from models import C2W, LanguageModel, W2C


def C2W2C(n_batch, params, V_C, c2w_trainable=True, lm_trainable=True, w2c_trainable=True):
  """
    Returns tuple (c2w2c model, sub-models, inputs)
  """
  # params
  maxlen    = params.maxlen
  d_C       = params.d_C
  d_Wi      = params.d_Wi
  d_W       = params.d_W
  d_L       = params.d_L
  d_D       = params.d_D

  # inputs
  w_nc      = Input(batch_shape=(n_batch, maxlen, V_C.size), name='w_nc', dtype='int8')
  w_nmask   = Input(batch_shape=(n_batch, 1), name='w_nmask', dtype='int8')
  w_np1c    = Input(batch_shape=(n_batch, maxlen, V_C.size), name='w_np1c', dtype='int8')

  # sub-models
  c2w       = C2W(maxlen, d_C, d_W, d_Wi, V_C, trainable=c2w_trainable)
  lm        = LanguageModel(n_batch, d_W, d_L, trainable=lm_trainable)
  w2c       = W2C(n_batch, maxlen, d_W, d_D, V_C, trainable=w2c_trainable)

  # the actual c2w2c model
  w_nE      = Dropout(.5)(c2w(w_nc))
  w_np1E    = Dropout(.5)(lm([w_nE, w_nmask]))
  w_np1     = Activation('softmax')(w2c([w_np1E, w_np1c]))
  c2w2c     = Model(input=[w_nc, w_nmask, w_np1c], output=w_np1)

  return c2w2c, (c2w, lm, w2c), (w_nc, w_nmask, w_np1c)


def build_c2w2c_validation_models(params, V_C):
  d_W       = params.d_W
  n_batch   = params.n_batch

  _, (c2w, lm, w2c), inputs = C2W2C(n_batch, params, V_C)
  w_nc, w_nmask, w_np1c     = inputs

  w_nE      = c2w(w_nc)
  w_np1E    = lm([w_nE, w_nmask])
  c2wp1     = Model(input=[w_nc, w_nmask], output=w_np1E)

  w_np1Ein  = Input(batch_shape=(n_batch, d_W), name='w_np1e', dtype='floatX')
  w_np1     = Activation('softmax')(w2c([w_np1Ein, w_np1c]))
  w2c       = Model(input=[w_np1Ein, w_np1c], output=w_np1)

  return c2wp1, w2c
