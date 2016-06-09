import numpy as np

from common import pad


def normalized(x):
  v = np.linalg.norm(x)
  if v == 0:
    return x
  return x / v


def calc_perplexity(V_W, V_Wt, V_C, expected, predictions):
  prob_sentence = 1.0
  for idx, word in enumerate(expected):
    probs = np.zeros(shape=(V_Wt.size,), dtype=np.float64)
    word_pred = predictions[idx]
    for tok in V_Wt.tokens:
      ptok = pad(tok, V_W)
      prob_word = 1.0
      for i, ch in enumerate(ptok):
        prob_word *= word_pred[i, V_C.get_index(ch)]
      probs[V_Wt.get_index(tok)] = prob_word
    probs = normalized(probs)
    prob_norm = probs[V_Wt.get_index(word)]
    prob_sentence *= prob_norm
  return np.power(prob_sentence, -1.0 / len(expected))
