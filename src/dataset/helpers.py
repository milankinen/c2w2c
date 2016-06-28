import numpy as np

from ..common import w2tok
from ..constants import SOW


def fill_word_one_hots(X, word, V_C, maxlen, pad=None):
  tok = w2tok(word, maxlen)
  for i, ch in enumerate(tok):
    if ch != SOW:
      X[i, V_C.get_index(ch)] = 1
  if pad is not None:
    for i in range(len(tok), maxlen):
      X[i, V_C.get_index(pad)] = 1


def fill_context_one_hots(X, ctx, V_C, maxlen, pad=None):
  n = len(ctx)
  for i in range(0, n):
    fill_word_one_hots(X[i], ctx[i], V_C, maxlen, pad)


def fill_weights(w, word, maxlen):
  tok = w2tok(word, maxlen)
  for i in range(0, len(tok)):
    w[i] = 1.


def hot2word(hots, V_C):
  return ['' if np.sum(h) == 0 else V_C.get_token(np.argmax(h)) for h in hots]
