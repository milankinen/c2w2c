import numpy as np

from ..datagen import prepare_data, to_c2w2w_samples
from helpers import calc_pp


def _calc_pp(c2w2w, meta, n_batch, generator, V_W):
  def loss_fn(P, expected):
    is_oov = not V_W.has(expected)
    if not is_oov:
      token_index = V_W.get_index(expected)
      word_loss   = -np.log(P[token_index] / np.sum(P))
      return word_loss

  return calc_pp(c2w2w, n_batch, meta, generator, loss_fn)


def make_c2w2w_test_function(c2w2w, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = to_c2w2w_samples(params, V_C, V_W)(samples)
    W_np1   = list([(None if s is None else s[1]) for s in samples])
    return X, W_np1

  n_batch                    = params.n_batch
  n_samples, meta, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_pp(c2w2w, meta, n_batch, generator, V_W)

