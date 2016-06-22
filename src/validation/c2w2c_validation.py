import numpy as np
import sys

from ..common import w2tok, w2str, is_oov
from ..constants import SOW, EOW
from ..dataset.helpers import fill_word_one_hots


def _get_word_probabilities(V_C, V_W, maxlen, pred):
  p_all = np.zeros(shape=(V_W.size,), dtype=np.float64)
  for word in V_W.tokens:
    tok = w2tok(word, maxlen, pad=EOW)
    p_all[V_W.get_index(word)] = np.prod([pred[i, V_C.get_index(ch)] for i, ch in enumerate(tok)])
  return p_all


def _calc_word_loss(expected, pred, V_C, V_W, maxlen):
  p_all = _get_word_probabilities(V_C, V_W, maxlen, pred)

  p = p_all[V_W.get_index(expected)] / np.sum(p_all)
  return -np.log(p)


def _calc_word_loss_quick(expected, pred, V_C, maxlen):
  def char_loss(i, ch):
    return -np.log(pred[i, V_C.get_index(ch)] / np.sum(pred[i]))
  tok = w2tok(expected, maxlen, pad=None)
  return np.sum([char_loss(i, ch) for i, ch in enumerate(tok)])


def _calc_perplexity(V_W, V_C, expectations, predictions, maxlen, quick_mode):
  tot_loss  = 0.
  n_oov     = 0
  n_tested  = 0
  for idx, expected in enumerate(expectations):
    if is_oov(expected, maxlen):
      n_oov += 1
      continue
    pred = predictions[idx]
    if quick_mode:
      word_loss = _calc_word_loss_quick(expected, pred, V_C, maxlen)
    else:
      word_loss = _calc_word_loss(expected, pred, V_C, V_W, maxlen)
    tot_loss += word_loss
    n_tested += 1

  return (0.0 if n_tested == 0 else np.exp(tot_loss / n_tested)), n_oov, n_tested


def _sample_word_predictions(w2c, W_np1e, maxlen, V_C):
  EOW_idx     = V_C.get_index(EOW)
  n_ctx       = len(W_np1e)
  predictions = np.zeros(shape=(n_ctx, maxlen, V_C.size), dtype=np.float32)
  for i in range(0, n_ctx):
    w_np1e  = W_np1e[i]
    w_np1e  = np.reshape(w_np1e, (1,) + w_np1e.shape)
    w_np1c  = np.zeros(shape=(1, maxlen, V_C.size), dtype=np.bool)
    fill_word_one_hots(w_np1c[0], SOW, V_C, maxlen, pad=EOW)
    for j in range(0, maxlen):
      step = w2c.predict({'w_np1e': w_np1e, 'w_np1c': w_np1c}, batch_size=1)[0]
      np.copyto(predictions[i, j], step[j])
      ch_idx = np.argmax(step[j])
      ch = V_C.get_token(ch_idx)
      if ch == EOW:
        # don't waste computation time because we stop word probability
        # calculation the when EOW character is encountered
        for k in range(j + 1, maxlen):
          np.copyto(predictions[i, k], step[k])
        return predictions
      elif j < maxlen - 1:
        # use predicted char as a sample for next character prediction
        w_np1c[0, j + 1, EOW_idx]  = 0
        w_np1c[0, j + 1, ch_idx]   = 1
  return predictions


def _test_model(params, lm, w2c, samples, V_W, V_C, quick_mode=False):
  maxlen        = params.maxlen
  total_samples = 0
  total_pp      = 0.0
  total_tested  = 0
  total_oov     = 0
  for expectations, X in samples:
    lm.reset_states()
    W_np1e          = lm.predict(X, batch_size=1)
    predictions     = _sample_word_predictions(w2c, W_np1e, maxlen, V_C)
    pp, oov, tested = _calc_perplexity(V_W, V_C, expectations, predictions, params.maxlen, quick_mode)
    if np.isnan(pp):
      # for some reason, S_e word predictions are randomly NaN => PP goes NaN
      # as a quick fix, just drop NaN sentences from the validation set for now
      if not quick_mode:
        print 'WARN failed to predict sentence PP: ' + ' '.join([w2str(e) for e in expectations])
      continue
    total_pp += pp
    total_tested += tested
    total_oov += oov
    total_samples += 1

  pp_avg   = sys.float_info.max if total_samples == 0 else total_pp / total_samples
  oov_rate = 0. if total_samples == 0 else 1.0 * total_oov / (total_oov + total_tested)
  return pp_avg, oov_rate


def make_c2w2c_test_function(lm, w2c, params, dataset, V_C, V_W):
  sents   = dataset.sentences
  maxlen  = params.maxlen
  samples = []
  for s in sents:
    n_ctx = len(s) - 1
    X = np.zeros(shape=(n_ctx, maxlen, V_C.size), dtype=np.bool)
    M = np.ones(shape=(n_ctx, 1), dtype=np.bool)
    for i in range(0, n_ctx):
      fill_word_one_hots(X[i], s[i], V_C, maxlen)
    samples.append((s[1:], {'w_nc': X, 'w_nmask': M}))

  def test_model(limit=None):
    if limit is None:
      return _test_model(params, lm, w2c, samples, V_W, V_C, quick_mode=False)
    else:
      return _test_model(params, lm, w2c, samples[0: min(len(samples), limit)], V_W, V_C, quick_mode=True)

  return test_model

