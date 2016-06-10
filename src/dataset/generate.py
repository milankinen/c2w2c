import random

import numpy as np

from constants import EOS, SOS
from dataset import Dataset


def _tok_str(tok):
  if tok == SOS:
    return '<S>'
  elif tok == EOS:
    return '</S>'
  return tok


def _fill_char_indices(X, word, V_C, maxlen):
  for i, ch in enumerate(word):
    if i >= maxlen:
      return
    X[i] = V_C.get_index(ch) + 1


def _fill_context_indices(X, ctx, V_C, maxlen):
  n = len(ctx)
  for i in range(0, n):
    _fill_char_indices(X[i], ctx[i], V_C, maxlen)


def _fill_char_one_hots(X, word, V_C, maxlen):
  for i, ch in enumerate(word):
    if i >= maxlen:
      return
    X[i, V_C.get_index(ch)] = 1


def make_training_samples_generator(params, dataset, V_C):
  sents = dataset.sentences[:]
  random.shuffle(sents)
  words     = Dataset(sents).get_words()
  n_words   = len(words)
  n_batch   = params.n_batch
  n_context = params.n_context
  n_samples = n_words - n_context - 1

  def make_generator():
    idx = 0
    while 1:
      actual_size = min(n_batch, n_words - idx - n_context - 1)
      X = np.zeros(shape=(actual_size, n_context, params.maxlen), dtype=np.int32)
      y = np.zeros(shape=(actual_size, params.maxlen, V_C.size), dtype=np.bool)
      for i in range(0, actual_size):
        _fill_context_indices(X[i], words[idx + i:idx + i + n_context], V_C, params.maxlen)
        _fill_char_one_hots(y[i], words[idx + i + n_context], V_C, params.maxlen)
      idx += actual_size
      if idx >= n_words:
        idx = 0
      """
      for a in range(0, actual_size):
        ctx = ''
        pred = ''
        for b in range(0, n_context):
          word = ''
          for c in range(0, params.maxlen):
            if X[a, b, c] == -1:
              break
            word += V_C.get_token(X[a, b, c])
          ctx += tok_str(word) + ' '
        for b in range(0, params.maxlen):
          for c in range(0, V_C.size):
            if y[a, b, c]:
              pred += V_C.get_token(c)
              break
        print ctx + ' >>> ' + pred
      """
      yield (X, y)

  return make_generator(), n_samples


def make_test_samples(params, dataset, V_C):
  sents = dataset.sentences
  X     = []
  for s in sents:
    x = np.zeros(shape=(1, len(s) - 1, params.maxlen), dtype=np.int32)
    _fill_context_indices(x[0], s[:-1], V_C, params.maxlen)
    X.append((s[1:], x))
  return X

