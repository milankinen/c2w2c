import numpy as np

from common import w2tok, w2str, is_oov


def _normalized(x):
  # == softmax
  return np.exp(x) / np.sum(np.exp(x), axis=0)


def _print_probability_distribution(expected, p_all, V_W):
  wp_pairs = [(w, p_all[V_W.get_index(w)]) for w in V_W.tokens]
  wp_pairs = sorted(wp_pairs, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)
  print '\n\nPROBABILITIES (EXPECTED: %s)' % w2str(expected)
  for w, p in wp_pairs:
    print '>> %s%s : %f' % ('*  ' if w == expected else '', w2str(w), p)


def calc_word_probability(word, pred, V_C, maxlen):
  p   = 0.
  tok = w2tok(word, maxlen)
  for i, ch in enumerate(tok):
    p += np.log(pred[i, V_C.get_index(ch)])
  # length normalization so that we don't favor short words
  return np.power(np.exp(p), 1.0 / len(tok))


def calc_perplexity(V_W, V_C, expectations, predictions, maxlen):
  p_sentence  = 1.0
  n_oov       = 0
  n_tested    = 0
  for idx, expected in enumerate(expectations):
    if is_oov(expected, maxlen):
      n_oov += 1
      continue
    pred  = predictions[idx]
    p_all = np.zeros(shape=(V_W.size,), dtype=np.float64)
    for word in V_W.tokens:
      p_word = calc_word_probability(word, pred, V_C, maxlen)
      p_all[V_W.get_index(word)] = p_word
    # normalize probabilities over the vocabulary
    p_all = _normalized(p_all)
    _print_probability_distribution(expected, p_all, V_W)
    p_expected = p_all[V_W.get_index(expected)]
    p_sentence *= p_expected
    n_tested += 1

  return (0.0 if n_tested == 0 else np.power(p_sentence, -1.0 / n_tested)), n_oov, n_tested


def test_model(params, lm, w2c, samples, V_W, V_C):
  total_pp      = 0.0
  total_tested  = 0
  total_oov     = 0
  for expected, x in samples:
    # S_e = predicted word embeddings that should match "expected"
    S_e = lm.predict(x)[0]
    pp, oov, tested = calc_perplexity(V_W, V_C, expected, w2c.predict(S_e), params.maxlen)
    total_pp += pp
    total_tested += tested
    total_oov += oov

  pp_avg   = total_pp / len(samples)
  oov_rate = 1.0 * total_oov / (total_oov + total_tested)
  return pp_avg, oov_rate
