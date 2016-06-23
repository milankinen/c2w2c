import random

import numpy as np

from constants import SOW, EOW, UNK
from dataset.helpers import fill_word_one_hots, fill_weights, fill_context_one_hots


def _groupby(col, fn):
  res = {}
  for v in col:
    k = fn(v)
    if not (k in res):
      res[k] = [v]
    else:
      res[k].append(v)
  return res


def _arrange_to_batches(sentences, n_batch):
  """
    This ugly piece of code arranges the given sentences into batches of n_batch
    so that sentences with same length and nearest length are grouped into same
    batches.

    The return value is list of batch tuples: (sentence_len_in_batch, [batch_sentence])
  """
  by_len    = _groupby(sentences, lambda s: len(s))
  by_len    = sorted(list((l, by_len[l]) for l in by_len), cmp=lambda a, b: cmp(a[0], b[0]), reverse=True)
  batches   = []
  b = []
  ml = 0
  for l, sents in by_len:
    if l > ml:
      ml = l
    for s in sents:
      b.append(s)
      if len(b) == n_batch:
        batches.append((ml, b))
        ml = 0
        b = []
  if 0 < len(b) < n_batch:
    b = b + [b[-1]] * (n_batch - len(b))
    batches.append((ml, b))
  return batches


def _shuffle_batches(batches):
  batches = batches[:]
  for i in range(0, len(batches)):
    b = (batches[i][0], batches[i][1])
    random.shuffle(b[1])
    batches[i] = b
  random.shuffle(batches)
  return batches


def _prepare_data(n_batch, dataset, to_samples, shuffle):
  sents     = dataset.sentences[:]
  n_samples = (dataset.n_words // n_batch) * n_batch

  def make_generator():
    while 1:
      if shuffle:
        random.shuffle(sents)
      samples = list([w for s in sents for w in s])

      for i in range(0, n_samples, n_batch):
        samples_x   = samples[i: i + n_batch]
        samples_y   = samples[i + 1: i + 1 + n_batch]
        yield to_samples(zip(samples_x, samples_y))

  return n_samples, make_generator()


def _to_c2w2c_samples(params, V_C):
  def to_samples(samples):
    n_batch = len(samples)
    maxlen  = params.maxlen
    X_nc    = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    X_nmask = np.zeros(shape=(n_batch, 1), dtype=np.bool)
    X_np1c  = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    y       = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    W       = np.zeros(shape=(n_batch, maxlen), dtype=np.float32)
    for i, sample in enumerate(samples):
      w_n, w_np1 = sample
      X_nmask[i, 0] = 1
      fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
      fill_word_one_hots(X_np1c[i], SOW + w_np1, V_C, maxlen, pad=EOW)
      fill_word_one_hots(y[i], w_np1, V_C, maxlen, pad=EOW)
      fill_weights(W[i], w_np1, maxlen)
    return {'w_nc': X_nc, 'w_nmask': X_nmask, 'w_np1c': X_np1c}, y, W
  return to_samples


def _to_c2w2w_samples(params, V_C, V_W):
  def to_samples(samples):
    n_batch = len(samples)
    maxlen  = params.maxlen
    X_nc    = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    X_nmask = np.zeros(shape=(n_batch, 1), dtype=np.bool)
    y       = np.zeros(shape=(n_batch, V_W.size), dtype=np.bool)
    W       = np.zeros(shape=(n_batch,), dtype=np.float32)
    for i, sample in enumerate(samples):
      w_n, w_np1 = sample
      y_idx      = V_W.get_index(w_np1 if V_W.has(w_np1) else UNK)
      fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
      X_nmask[i, 0] = 1
      y[i, y_idx]   = 1
      W[i]          = 1.
    return {'w_nc': X_nc, 'w_nmask': X_nmask}, y, W
  return to_samples


def prepare_c2w2c_training_data(params, dataset, V_C, shuffle=True):
  return _prepare_data(params.n_batch, dataset, _to_c2w2c_samples(params, V_C), shuffle)


def prepare_c2w2w_training_data(params, dataset, V_C, V_W, shuffle=True):
  return _prepare_data(params.n_batch, dataset, _to_c2w2w_samples(params, V_C, V_W), shuffle)


def preprare_w2c_training_data(params, dataset, V_C, V_W, c2w2w):
  n_batch   = params.n_batch
  maxlen    = params.maxlen
  sents     = dataset.sentences
  n_samples = sum([len(s) - 1 for s in sents])
  Xo = np.zeros(shape=(n_samples, params.d_W), dtype=np.float32)
  yo = []
  i = 0
  for s in sents:
    n_ctx = len(s) - 1
    c2w2w.reset_states()
    Xw = np.zeros(shape=(n_ctx, maxlen, V_C.size), dtype=np.bool)
    Xm = np.ones(shape=(n_ctx, ), dtype=np.bool)
    fill_context_one_hots(Xw, s[0:-1], V_C, maxlen)
    x = c2w2w.predict({'w_nc': Xw, 'w_nmask': Xm}, batch_size=1)
    for j in range(0, n_ctx):
      np.copyto(Xo[i], x[j])
      yo.append(s[j + 1])
      i += 1

  assert n_samples == len(yo)

  def make_generator():
    # shuffle data before iteration
    p = np.random.permutation(n_samples)
    Xs = Xo[p]
    ys = []
    for i in p: ys.append(yo[i])

    def generator_fn():
      idx = 0
      while 1:
        actual = min(n_samples - idx, n_batch)
        Xe  = np.zeros(shape=(actual, params.d_W), dtype=np.float32)
        Xc  = np.zeros(shape=(actual, maxlen, V_C.size), dtype=np.bool)
        y   = np.zeros(shape=(actual, maxlen, V_C.size), dtype=np.bool)
        W   = np.zeros(shape=(actual, maxlen), dtype=np.float32)
        for i in range(0, actual):
          np.copyto(Xe[i], Xs[idx])
          fill_word_one_hots(Xc[i], SOW + ys[idx], V_C, maxlen, pad=EOW)
          fill_word_one_hots(y[i], ys[idx], V_C, maxlen)
          fill_weights(W[i], ys[idx], dataset, maxlen)
          idx += 1
        if idx >= n_samples:
          idx = 0
        yield ({'w_np1e': Xe, 'w_np1c': Xc}, y, W)

    return [n_samples], generator_fn()

  return (n_samples, make_generator), (Xo, yo)

