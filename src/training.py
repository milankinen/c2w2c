import random

import numpy as np

from constants import SOW, EOW
from dataset.helpers import fill_word_one_hots, fill_weights


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


def _sample_generator(samples_to_vectors, batches, n_batch):
  i = 0
  j = 0
  while 1:
    samples = []
    n, b = batches[i]
    for k in range(0, n_batch):
      sent  = b[k]
      if len(sent) > j + 1:
        w_t   = sent[j]
        w_tp1 = sent[j + 1]
        samples.append((w_t, w_tp1))
      else:
        samples.append(None)
    sample_vectors = samples_to_vectors(samples)
    j += 1
    if j >= n - 1:
      j = 0
      i += 1
    if i >= len(batches):
      i = 0
      j = 0
    yield sample_vectors


def prepare_c2w2c_training_data(params, dataset, V_C):
  n_batch   = params.n_batch
  maxlen    = params.maxlen
  batches   = _arrange_to_batches(dataset.sentences, n_batch)
  n_samples = sum((b[0] - 1) * n_batch for b in batches)

  def samples_to_vectors(samples):
    X_nc    = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    X_nmask = np.zeros(shape=(n_batch, 1), dtype=np.bool)
    X_np1c  = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    y       = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    W       = np.zeros(shape=(n_batch, maxlen), dtype=np.float32)
    for i, sample in enumerate(samples):
      if sample is not None:
        w_n, w_np1 = sample
        X_nmask[i, 0] = 1
        fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
        fill_word_one_hots(X_np1c[i], SOW + w_np1, V_C, maxlen, pad=EOW)
        fill_word_one_hots(y[i], w_np1, V_C, maxlen, pad=EOW)
        fill_weights(W[i], w_np1, dataset, maxlen)
    return {'w_nc': X_nc, 'w_nmask': X_nmask, 'w_np1c': X_np1c}, y, W

  def make_generator():
    shuffled  = _shuffle_batches(batches)
    lengths   = list((b[0] - 1) for b in shuffled)
    generator = _sample_generator(samples_to_vectors, shuffled, n_batch)
    return lengths, generator

  return n_samples, make_generator


def prepare_c2w2w_training_data(params, dataset, V_C, V_W):
  n_batch   = params.n_batch
  maxlen    = params.maxlen
  batches   = _arrange_to_batches(dataset.sentences, n_batch)
  n_samples = sum((b[0] - 1) * n_batch for b in batches)

  def samples_to_vectors(samples):
    X_nc    = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    X_nmask = np.zeros(shape=(n_batch, 1), dtype=np.bool)
    y       = np.zeros(shape=(n_batch, V_W.size), dtype=np.bool)
    W       = np.zeros(shape=(n_batch,), dtype=np.float32)
    for i, sample in enumerate(samples):
      if sample is not None:
        w_n, w_np1 = sample
        fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
        X_nmask[i, 0]               = 1
        W[i]                        = 1.
        y[i, V_W.get_index(w_np1)]  = 1
    assert np.sum(X_nmask) > 0
    assert np.sum(y) > 0
    assert np.sum(W) > 0
    return {'w_nc': X_nc, 'w_nmask': X_nmask}, y, W

  def make_generator():
    shuffled  = _shuffle_batches(batches)
    lengths   = list((b[0] - 1) for b in shuffled)
    print lengths
    generator = _sample_generator(samples_to_vectors, shuffled, n_batch)
    return lengths, generator

  return n_samples, make_generator
