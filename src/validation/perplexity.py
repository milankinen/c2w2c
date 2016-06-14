import numpy as np

from common import w2tok, w2str, is_oov
from constants import SOW, EOW
from dataset.generate import _fill_char_one_hots


def _print_probability_distribution(expected, p_all, V_W):
  p_norm    = p_all / np.sum(p_all)
  wp_pairs  = [(w, p_norm[V_W.get_index(w)]) for w in V_W.tokens]
  wp_pairs  = sorted(wp_pairs, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)
  print '\n\nPROBABILITIES (EXPECTED: %s)' % w2str(expected)
  for w, p in wp_pairs:
    print '>> %s%s : %f' % ('*  ' if w == expected else '', w2str(w), p)


def calc_word_loss(word, pred, V_C, maxlen):
  loss = 0.
  tok = w2tok(word, maxlen)
  for i, ch in enumerate(tok):
    p_ch = pred[i]
    loss += -np.log(p_ch[V_C.get_index(tok[i])]) / np.sum(p_ch)
  return loss / len(tok)


def calc_perplexity(V_W, V_C, expectations, predictions, maxlen):
  tot_loss  = 0.
  n_oov     = 0
  n_tested  = 0
  for idx, expected in enumerate(expectations):
    if is_oov(expected, maxlen):
      n_oov += 1
      continue
    pred  = predictions[idx]
    word_loss = calc_word_loss(expected, pred, V_C, maxlen)
    tot_loss += word_loss
    n_tested += 1

  return (0.0 if n_tested == 0 else np.exp(tot_loss / n_tested)), n_oov, n_tested


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
  total_samples = 0
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
    if np.isnan(pp):
      # for some reason, S_e word predictions are randomly NaN => PP goes NaN
      # as a quick fix, just drop NaN sentences from the validation set for now
      print 'WARN failed to predict sentence PP: ' + ' '.join([w2str(e) for e in expectations])
      continue

    total_pp += pp
    total_tested += tested
    total_oov += oov
    total_samples += 1

  pp_avg   = total_pp / total_samples
  oov_rate = 1.0 * total_oov / (total_oov + total_tested)
  return pp_avg, oov_rate
