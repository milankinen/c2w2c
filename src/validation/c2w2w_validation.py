import numpy as np
import sys

from ..constants import UNK
from ..training import _prepare_data, _to_c2w2w_samples


def _calc_perplexity(V_W, expectations, predictions):
  tot_loss  = 0.
  n_oov     = 0
  n_tested  = 0
  for idx, expected in enumerate(expectations):
    if not V_W.has(expected):
      n_oov += 1
      continue
    P_all     = predictions[idx]
    word_loss = -np.log(P_all[V_W.get_index(expected)] / np.sum(P_all))
    tot_loss += word_loss
    n_tested += 1

  return (0.0 if n_tested == 0 else np.exp(tot_loss / n_tested)), n_oov, n_tested


def _calc_loss(V_W, predictions, expectations):
  l, o, t = 0, 0, 0
  for idx, sample in enumerate(zip(predictions, expectations)):
    P, expected = sample
    is_oov      = not V_W.has(expected)
    token_index = V_W.get_index(UNK if is_oov else expected)
    word_loss   = -np.log(P[token_index] / np.sum(P))
    if np.isnan(word_loss):
      print 'WARN: unable to get loss of word: ' + expected
      o += 1
      continue
    l += word_loss
    o += 1 if is_oov else 0
    t += 0 if is_oov else 1

  #print 'loss', l
  return l, o, t


def _test_model(c2w2w, n_samples, n_batch, generator, V_W):
  l, o, t, n = 0., 0, 0, 0
  c2w2w.reset_states()
  while n < n_samples:
    X, expectations   = generator.next()
    predictions       = c2w2w.predict(X, batch_size=n_batch)
    loss, oov, tested = _calc_loss(V_W, predictions, expectations)
    l += loss
    o += oov
    t += tested
    n += len(expectations)

  assert n == n_samples

  pp    = sys.float_info.max if t == 0 else np.exp(l / t)
  oovr  = 0 if t + o == 0 else o / float(t + o)
  return pp, oovr


def make_c2w2w_test_function(c2w2w, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = _to_c2w2w_samples(params, V_C, V_W)(samples)
    W_np1   = list([s[1] for s in samples])
    return X, W_np1

  n_batch = params.n_batch
  n_samples, generator = _prepare_data(n_batch, dataset, to_test_samples, shuffle=False)

  def test_model(limit=None):
    if limit is not None:
      print 'WARN: limit is not used anymore'
    return _test_model(c2w2w, n_samples, n_batch, generator, V_W)

  return test_model

