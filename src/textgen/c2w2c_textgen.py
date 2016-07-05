import sys

import numpy as np

from ..common import EOS, EOW, w2str
from ..dataset import initialize_c2w2c_data, Dataset


def _stdwrite(word):
  sys.stdout.write(w2str(word) + ' ')
  sys.stdout.flush()


def _sample_p_chars(model, c, maxlen, V_C):
  p_chars = np.zeros(shape=(maxlen, V_C.size), dtype=np.float32)
  y_tm1   = np.zeros(shape=(maxlen,), dtype=np.int32)
  for i in range(maxlen):
    p = model.predict_chars(c, y_tm1)[i]
    np.copyto(p_chars[i], p)
    next_ch = np.argmax(p)
    if i < maxlen - 1:
      y_tm1[i + 1] = next_ch + 1

  return p_chars


def _sample_word(model, c, maxlen, V_C):
  p_chars = _sample_p_chars(model, c, maxlen, V_C)
  word = ''
  for p in p_chars:
    ch = V_C.get_token(np.argmax(p))
    if ch == EOW:
      break
    word += ch
  return word


def generate_c2w2c_text(model, maxlen, seed, how_many):
  hyper_params    = model.get_hyperparams()
  batch_size, V_C = hyper_params[0], hyper_params[-1]

  def _step(words):
    sents = [words + [EOS]] * batch_size
    make_gen, _ = initialize_c2w2c_data(Dataset(sents), batch_size, maxlen, V_C, shuffle=False)
    n_samples, gen = make_gen()
    ctx_pred = model.predict_ctx(gen, n_samples)
    return ctx_pred[-1]

  print ''
  model.reset_states()
  for _ in range(how_many):
    c = _step(seed)
    next_word = _sample_word(model, c, maxlen, V_C)
    _stdwrite(next_word)
    seed = [next_word]

