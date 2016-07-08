import numpy as np
from keras.layers import Input, Dropout, Activation
from keras.models import Model

from .layers import ContextMask
from .sub_models import C2W, LanguageModel, W2C


def C2W2C(batch_size, maxlen, d_C, d_Wi, d_W, d_L, d_D, V_C):
  # inputs
  context   = Input(batch_shape=(batch_size, maxlen), name='context', dtype='int32')
  y_tm1     = Input(batch_shape=(batch_size, maxlen), name='y_tm1', dtype='int32')
  c         = Input(batch_shape=(batch_size, d_L), name='c', dtype='floatX')

  # sub-models
  c2w       = C2W(maxlen, d_C, d_W, d_Wi, V_C)
  lm        = LanguageModel(batch_size, d_W, d_L)
  w2c       = W2C(maxlen, d_L, d_C, d_D, V_C)

  # the actual c2w2c model
  ctx_mask  = ContextMask()(context)
  ctx_emb   = Dropout(.5)(c2w(context))
  C         = Dropout(.5)(lm([ctx_emb, ctx_mask]))
  y_logit   = w2c([C, y_tm1])
  y         = Activation('softmax')(y_logit)
  c2w2c     = Model(input=[context, y_tm1], output=y)

  # separe W2C for text generation
  w2c_logit = w2c([c, y_tm1])
  w2c_model = Model(input=[c, y_tm1], output=Activation('softmax')(w2c_logit))
  lm_model  = Model(input=context, output=C)

  def predict_ctx(gen, n_samples):
    return lm_model.predict_generator(gen, n_samples)

  def predict_chars(c_dat, y_tm1_dat):
    assert len(c_dat.shape) == len(y_tm1_dat.shape), (c_dat.shape, y_tm1_dat.shape)
    assert len(c_dat.shape) in [1, 2]
    assert c_dat.shape[-1] == d_L, c_dat.shape
    assert y_tm1_dat.shape[-1] == maxlen, y_tm1_dat.shape

    actual_size, n_samples = batch_size, 1
    if len(c_dat.shape) == 2:
      n_samples = c_dat.shape[0]
      actual_size = max(batch_size, n_samples)

    # stateful LSTMs require fixed-size batch so let's just pad other
    # samples with zeros
    C_dat = np.zeros((actual_size, d_L), dtype=np.float32)
    Y_tm1_dat = np.zeros((actual_size, maxlen), dtype=np.float32)
    if len(c_dat.shape) == 1:
      c_dat = np.reshape(c_dat, (1,) + c_dat.shape)
      y_tm1_dat = np.reshape(y_tm1_dat, (1,) + y_tm1_dat.shape)
    np.copyto(C_dat[0:n_samples], c_dat)
    np.copyto(Y_tm1_dat[0:n_samples], y_tm1_dat)
    return w2c_model.predict({'c': C_dat, 'y_tm1': Y_tm1_dat}, batch_size=batch_size)

  c2w2c.get_hyperparams = lambda: (batch_size, d_C, d_Wi, d_W, d_L, d_D, V_C)
  c2w2c.get_c2w = lambda: c2w
  c2w2c.get_lm = lambda: lm
  c2w2c.get_w2c = lambda: w2c
  c2w2c.predict_ctx = predict_ctx
  c2w2c.predict_chars = predict_chars

  return c2w2c

