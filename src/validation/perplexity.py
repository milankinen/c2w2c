import numpy as np

from dataset.generate import _tok_str


def _normalized(x):
  v = np.linalg.norm(x)
  if v == 0:
    return x
  return x / v


def _get_normalized_prob(x, idx):
  p = x[idx]
  return p / np.sum(x)


def _print_probs(expected, probs, V_W):
  pl = []
  for tok in V_W.tokens:
    pl.append((tok, probs[V_W.get_index(tok)]))
  pl = sorted(pl, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)
  print '\n\nPROBABILITIES (EXPECTED: %s)' % _tok_str(expected)
  for tok, p in pl:
    print '>> %s%s : %f' % ('*  ' if tok == expected else '', _tok_str(tok), p)


def calc_perplexity(V_W, V_C, expected, predictions, maxlen):
  prob_sent = 1.0
  oov       = 0
  tested    = 0
  for idx, word in enumerate(expected):
    if len(word) > maxlen:
      oov += 1
      continue
    probs = np.zeros(shape=(V_W.size,), dtype=np.float64)
    word_pred = predictions[idx]
    for tok in V_W.tokens:
      prob_tok = 1.0
      for i, ch in enumerate(tok):
        prob_tok *= word_pred[i, V_C.get_index(ch)]
      # length normalization so that we don't favor short words
      prob_tok = np.power(prob_tok, 1.0 / len(tok))
      probs[V_W.get_index(tok)] = prob_tok
    # normalize probabilities over the vocabulary
    probs = _normalized(probs)
    # _print_probs(word, probs, V_W)
    prob_norm = probs[V_W.get_index(word)] #_get_normalized_prob(probs, V_W.get_index(word))
    prob_sent *= prob_norm
    tested += 1

  return (0.0 if tested == 0 else np.power(prob_sent, -1.0 / tested)), oov, tested


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
