import sys, numpy as np

from ..common import w2str
from ..dataset.helpers import fill_word_one_hots


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


def _test_model(c2w2w, reset_lm, samples, V_W, quick_mode=False):
  total_samples = 0
  total_pp      = 0.0
  total_tested  = 0
  total_oov     = 0
  for expectations, X in samples:
    reset_lm()
    predictions     = c2w2w.predict(X, batch_size=1)
    pp, oov, tested = _calc_perplexity(V_W, expectations, predictions)
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


def make_c2w2w_test_function(c2w2w, reset_lm, params, dataset, V_C, V_W):
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
      return _test_model(c2w2w, reset_lm, samples, V_W, quick_mode=False)
    else:
      return _test_model(c2w2w, reset_lm, samples[0: min(len(samples), limit)], V_W, quick_mode=True)

  return test_model

