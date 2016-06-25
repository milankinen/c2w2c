import random

import numpy as np

from ..constants import SOW, EOW, UNK
from ..dataset.helpers import fill_word_one_hots, fill_weights


def prepare_data(n_batch, dataset, to_samples, shuffle):
  sents     = dataset.sentences[:]
  n_context = (dataset.n_words - 1) // n_batch
  n_samples = n_context * n_batch

  def make_generator():
    while 1:
      if shuffle:
        random.shuffle(sents)
      samples = list([w for s in sents for w in s])

      for i in range(0, n_context):
        X, y = [], []
        for j in range(0, n_batch):
          X.append(samples[j * n_context + i])
          y.append(samples[j * n_context + i + 1])
        yield to_samples(zip(X, y))

  return n_samples, make_generator()


def to_c2w2c_samples(params, V_C):
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


def to_c2w2w_samples(params, V_C, V_W):
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
  return prepare_data(params.n_batch, dataset, to_c2w2c_samples(params, V_C), shuffle)


def prepare_c2w2w_training_data(params, dataset, V_C, V_W, shuffle=True):
  return prepare_data(params.n_batch, dataset, to_c2w2w_samples(params, V_C, V_W), shuffle)


def prepare_w2c_training_data(c2wnp1, params, dataset, V_C):
  def _to_cached_w2c_samples(cache):
    def to_samples(samples):
      cache_key = ':'.join([s[0] for s in samples])
      if cache_key in cache:
        return cache[cache_key]
      # samples not found from cache, must generate embedding first
      X, y, W = to_c2w2c_samples(params, V_C)(samples)
      W_np1e  = c2wnp1.predict({'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}, batch_size=params.n_batch)
      samples = ({'w_np1e': W_np1e, 'w_np1c': X['w_np1c']}, y, W)
      cache[cache_key] = samples
      return samples
    return to_samples

  return prepare_data(params.n_batch, dataset, _to_cached_w2c_samples({}), shuffle=False)
