import numpy as np

from common import w2tok, w2str, is_oov
from constants import SOW, EOW
from dataset.generate import _fill_char_one_hots


# Softmax function without floating point overflow, see:
# https://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
def softmax(x):
  a = x.max()
  e = np.exp(x - a)
  z = np.sum(e)
  return e / z


def _print_probability_distribution(expected, p_all, V_W):
  p_norm    = p_all / np.sum(p_all)
  wp_pairs  = [(w, p_norm[V_W.get_index(w)]) for w in V_W.tokens]
  wp_pairs  = sorted(wp_pairs, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)
  print '\n\nPROBABILITIES (EXPECTED: %s)' % w2str(expected)
  for w, p in wp_pairs:
    print '>> %s%s : %f' % ('*  ' if w == expected else '', w2str(w), p)


def calc_word_probability(word, pred, V_C, maxlen):
  p   = 0.
  tok = w2tok(word, maxlen)
  for i, ch in enumerate(tok):
    p += np.log(pred[i, V_C.get_index(tok[i])])
  # length normalization so that we don't favor short words
  return np.power(np.exp(p), 1.0 / len(tok))


def get_word_probability(word, p_all, V_W):
  p = p_all[V_W.get_index(word)]
  return p / np.sum(p_all)


def calc_perplexity(V_W, V_C, expectations, predictions, maxlen):
  p_sentence  = 0.
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
    #_print_probability_distribution(expected, p_all, V_W)
    p_expected = get_word_probability(expected, p_all, V_W)
    p_sentence += np.log(p_expected)
    n_tested += 1

  return (0.0 if n_tested == 0 else np.power(np.exp(p_sentence), -1.0 / n_tested)), n_oov, n_tested


def sample_word_prediction_to(target, w2c, embedding, maxlen, V_C):
  EOW_idx = V_C.get_index(EOW)
  Xe      = np.reshape(embedding, (1,) + embedding.shape)
  Xword   = np.zeros(shape=(1, maxlen, V_C.size), dtype=np.bool)
  _fill_char_one_hots(Xword[0], SOW, V_C, maxlen, pad=EOW)
  for i in range(0, maxlen):
    step = w2c.predict({'embedding': Xe, 'predicted_word': Xword})[0]
    np.copyto(target[i], step[i])
    ch_idx = np.argmax(step[i])
    ch = V_C.get_token(ch_idx)
    if ch == EOW:
      # don't waste computation time because we stop word probability
      # calculation the when EOW character is encountered
      for j in range(i + 1, maxlen):
        np.copyto(target[j], step[j])
      return
    elif i < maxlen - 1:
      # use predicted char as a sample for next character prediction
      Xword[0, i + 1, EOW_idx]  = 0
      Xword[0, i + 1, ch_idx]   = 1


def test_model(params, lm, w2c, samples, V_W, V_C):
  maxlen        = params.maxlen
  total_pp      = 0.0
  total_tested  = 0
  total_oov     = 0
  for expectations, X in samples:
    n_words = len(expectations)
    # S_e = predicted word embeddings that should match "expected"
    S_e = lm.predict(X, batch_size=n_words)
    predictions = np.zeros(shape=(n_words, maxlen, V_C.size), dtype=np.float64)
    for i in range(0, n_words):
      sample_word_prediction_to(predictions[i], w2c, S_e[i], maxlen, V_C)
    pp, oov, tested = calc_perplexity(V_W, V_C, expectations, predictions, params.maxlen)
    total_pp += pp
    total_tested += tested
    total_oov += oov

  pp_avg   = total_pp / len(samples)
  oov_rate = 1.0 * total_oov / (total_oov + total_tested)
  return pp_avg, oov_rate
