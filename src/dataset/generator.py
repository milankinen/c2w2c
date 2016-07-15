import random

import numpy as np

from ..common import w2tok
from ..constants import EOW, UNK


def _grouped(col, n):
  i, l, dst = 0, len(col), []
  while i < l:
    dst.append(col[i: i + min(l - i, n)])
    i += len(dst[-1])
  return dst


def make_generator(batch_size, dataset, to_samples, shuffle):
  max_words = max(dataset.n_words // batch_size, 2)
  sentences = dataset.sentences[:]
  if shuffle:
    random.shuffle(sentences)
  data_rows = _grouped([w for s in sentences for w in s], max_words)[0: batch_size]
  data_rows += [[]] * max(0, (batch_size - len(data_rows)))
  assert len(data_rows) == batch_size, str(len(data_rows))
  num_batches = len(data_rows[0]) - 1

  def generator():
    while 1:
      for i in range(num_batches):
        batch = []
        for row in data_rows:
          batch.append((row[i], row[i + 1]) if i < len(row) - 1 else None)
        assert any([any(samples) for samples in batch]), str(batch)
        yield to_samples(batch)

  n_samples = num_batches * batch_size
  return n_samples, generator()


def initialize_c2w2c_data(dataset, batch_size, maxlen, V_C, shuffle=True, include_oov=True):
  cache = {}
  for word in dataset.vocabulary.tokens:
    is_oov  = False
    token   = w2tok(word, maxlen, pad=None)
    chars   = np.zeros(shape=(maxlen,), dtype=np.int32)
    n_chars = 0
    if len(word) + 1 > maxlen:
      is_oov = True
    for ch in token:
      if V_C.has(ch):
        chars[n_chars] = V_C.get_index(ch) + 1
        n_chars += 1
      else:
        is_oov = True
    for i in range(n_chars, maxlen):
      chars[i] = V_C.get_index(EOW) + 1
    weights = np.array([1.] * n_chars + [0.] * (maxlen - n_chars), dtype=np.float32)
    cache[word] = chars, weights, n_chars, is_oov

  def to_samples(batch):
    ctx   = np.zeros(shape=(batch_size, maxlen), dtype=np.int32)
    y_tm1 = np.zeros(shape=(batch_size, maxlen), dtype=np.int32)
    y     = np.zeros(shape=(batch_size, maxlen), dtype=np.int32)
    y_w   = np.zeros(shape=(batch_size, maxlen), dtype=np.float32)
    for i, sample in enumerate(batch):
      if sample is not None:
        w_t, w_tp1         = sample
        x_chars, _, n, _   = cache[w_t]
        y_chars, w, _, oov = cache[w_tp1]
        for k in range(n):
          ctx[i, k] = x_chars[k]
        np.copyto(y_tm1[i][1:], y_chars[0: -1])
        np.copyto(y[i], y_chars - 1)
        if oov and not include_oov:
          continue
        np.copyto(y_w[i], w)
    # sparse_categorical_crossentropy requires y to have the same shape as model output
    y = np.expand_dims(y, -1)
    return {'context': ctx, 'y_tm1': y_tm1}, y, y_w

  def _make():
    return make_generator(batch_size, dataset, to_samples, shuffle)

  n_oov     = sum((1 if cache[w][3] else 0) for w in dataset.get_words())
  oov_rate  = 1. * n_oov / dataset.n_words

  return _make, oov_rate


def initialize_word_lstm_data(dataset, batch_size, V_W, shuffle=True, include_oov=True):
  def is_oov(w):
    return not V_W.has(w)

  def to_samples(batch):
    ctx   = np.zeros(shape=(batch_size, V_W.size), dtype=np.bool)
    y     = np.zeros(shape=(batch_size,), dtype=np.int32)
    y_w   = np.zeros(shape=(batch_size,), dtype=np.float32)
    for i, sample in enumerate(batch):
      if sample is not None:
        w_t, w_tp1 = sample
        w_t     = UNK if is_oov(w_t) else w_t
        w_tp1   = UNK if is_oov(w_tp1) else w_tp1
        y[i]    = V_W.get_index(w_tp1)
        ctx[i, V_W.get_index(w_t)] = 1
        if w_tp1 == UNK and not include_oov:
          continue
        y_w[i]  = 1.

    # sparse_categorical_crossentropy requires y to have the same shape as model output
    y = np.expand_dims(y, -1)
    return ctx, y, y_w

  def _make():
    return make_generator(batch_size, dataset, to_samples, shuffle)

  n_oov     = sum((1 if is_oov(w) else 0) for w in dataset.get_words())
  oov_rate  = 1. * n_oov / dataset.n_words

  return _make, oov_rate
