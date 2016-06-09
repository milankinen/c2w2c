import numpy as np

from constants import SOS, EOS


def normalized(x):
  v = np.linalg.norm(x)
  if v == 0:
    return x
  return x / v


def tok_str(tok):
  if tok == SOS:
    return '<S>'
  elif tok == EOS:
    return '</S>'
  return tok


def print_probs(expected, probs, V_Wt):
  pl = []
  for tok in V_Wt.tokens:
    pl.append((tok, probs[V_Wt.get_index(tok)]))
  pl = sorted(pl, cmp=lambda a, b: cmp(a[1], b[1]), reverse=True)
  print '\n\nPROBABILITIES (EXPECTED: %s)' % tok_str(expected)
  for tok, p in pl:
    print '>> %s%s : %f' % ('*  ' if tok == expected else '', tok_str(tok), p)


def calc_perplexity(V_W, V_Wt, V_C, expected, predictions):
  prob_sentence = 1.0
  for idx, word in enumerate(expected):
    probs = np.zeros(shape=(V_Wt.size,), dtype=np.float64)
    word_pred = predictions[idx]
    #print '\n\n' + word
    #print word_pred
    for tok in V_Wt.tokens:
      prob_tok = 1.0
      for i, ch in enumerate(tok):
        prob_tok *= word_pred[i, V_C.get_index(ch)]
      # length normalization so that we don't favor short words
      prob_tok = np.power(prob_tok, 1.0 / len(tok))
      probs[V_Wt.get_index(tok)] = prob_tok
    # normalize probabilities over the vocabulary
    probs = normalized(probs)
    print_probs(word, probs, V_Wt)
    prob_norm = probs[V_Wt.get_index(word)]
    prob_sentence *= prob_norm
  return np.power(prob_sentence, -1.0 / len(expected))
