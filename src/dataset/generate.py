import random

import numpy as np

from common import w2tok
from constants import EOW
from dataset import Dataset


def _fill_char_one_hots(X, word, V_C, maxlen, pad=None):
  tok = w2tok(word, maxlen)
  for i, ch in enumerate(tok):
    X[i, V_C.get_index(ch)] = 1
  if pad is not None:
    for i in range(len(tok), maxlen):
      X[i, V_C.get_index(pad)] = 1


def _fill_context_one_hots(X, ctx, V_C, maxlen, pad=None):
  n = len(ctx)
  for i in range(0, n):
    _fill_char_one_hots(X[i], ctx[i], V_C, maxlen, pad)


def _fill_weights(w, word, maxlen):
  tok = w2tok(word, maxlen)
  for i in range(0, len(tok)):
    w[i] = 1.


def make_training_samples_generator(params, dataset, V_C):
  sents = dataset.sentences[:]
  random.shuffle(sents)
  maxlen    = params.maxlen
  words     = Dataset(sents).get_words()
  n_words   = len(words)
  n_batch   = params.n_batch
  n_context = params.n_context
  n_samples = n_words - n_context - 1

  def make_generator():
    idx = 0
    while 1:
      actual_size = min(n_batch, n_words - idx - n_context - 1)
      X = np.zeros(shape=(actual_size, n_context, maxlen, V_C.size), dtype=np.bool)
      y = np.zeros(shape=(actual_size, maxlen, V_C.size), dtype=np.bool)
      w = np.zeros(shape=(actual_size, maxlen), dtype=np.float32)
      for i in range(0, actual_size):
        _fill_context_one_hots(X[i], words[idx + i:idx + i + n_context], V_C, maxlen)
        _fill_char_one_hots(y[i], words[idx + i + n_context], V_C, maxlen, pad=EOW)
        _fill_weights(w[i], words[idx + i + n_context], maxlen)
      idx += actual_size
      if idx >= n_words:
        idx = 0
      yield (X, y, w)

  return make_generator(), n_samples


def make_test_samples(params, dataset, V_C):
  sents   = dataset.sentences
  maxlen  = params.maxlen
  X       = []
  for s in sents:
    x = np.zeros(shape=(1, len(s) - 1, maxlen, V_C.size), dtype=np.bool)
    _fill_context_one_hots(x[0], s[:-1], V_C, maxlen)
    X.append((s[1:], x))
  return X

