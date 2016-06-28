import numpy as np

from ..constants import SOW, EOW, EOS, UNK
from ..dataset.helpers import fill_word_one_hots, fill_weights


def _grouped(col, n):
  i, l, dst = 0, len(col), []
  while i < l:
    dst.append(col[i: i + min(l - i, n)])
    i += len(dst[-1])
  return dst


def _batches_per_sentence(n_batch, dataset):
  sents = sorted(dataset.sentences[:], cmp=lambda a, b: cmp(len(a), len(b)), reverse=True)
  if len(sents) % n_batch != 0:
    sents += [None] * (n_batch - (len(sents) % n_batch))
  assert len(sents) % n_batch == 0
  batches   = _grouped(sents, n_batch)
  meta      = list(len(b[0]) - 1 for b in batches)
  return batches, meta


def _batches_per_dataset(n_batch, dataset):
  sents = dataset.sentences[:]
  n_max = dataset.n_words // n_batch
  batch = [[]]
  for s in sents:
    n_new = len(s) + len(batch[-1])
    if n_new > n_max:
      batch.append([])
    batch[-1] += s
  batch = sorted(batch[0: n_batch], cmp=lambda a, b: cmp(len(a), len(b)), reverse=True)
  meta  = len(batch[0]) - 1
  return [batch], [meta]


def prepare_data(n_batch, dataset, to_samples, shuffle):
  #batches, meta = _batches_per_sentence(n_batch, dataset)
  batches, meta = _batches_per_dataset(n_batch, dataset)
  n_samples     = sum(meta) * n_batch

  if shuffle:
    b, m = [], []
    p = np.random.permutation(len(batches))
    for i in p:
      m.append(meta[i])
      b.append(batches[i])
    meta    = m
    batches = b

  def generator():
    while 1:
      for batch, n in zip(batches, meta):
        for i in range(0, n):
          samples = []
          for b in batch:
            samples.append((b[i], b[i + 1]) if b is not None and i < len(b) - 1 else None)
          yield to_samples(samples)

  return n_samples, meta, generator()


def to_c2w2c_samples(params, V_C):
  def to_samples(samples):
    n_batch = len(samples)
    maxlen  = params.maxlen
    X_nc    = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    X_nmask = np.zeros(shape=(n_batch, 1), dtype=np.bool)
    X_np1c  = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    y       = np.zeros(shape=(n_batch, maxlen, V_C.size), dtype=np.bool)
    W       = np.ones(shape=(n_batch, maxlen), dtype=np.float32)
    for i, sample in enumerate(samples):
      if sample is not None:
        w_n, w_np1 = sample
        X_nmask[i, 0] = 1
        fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
        fill_word_one_hots(X_np1c[i], SOW + w_np1, V_C, maxlen, pad=EOW)
        fill_word_one_hots(y[i], w_np1, V_C, maxlen, pad=EOW)
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
      if sample is not None:
        w_n, w_np1 = sample
        w_np1 = w_np1 if V_W.has(w_np1) else UNK
        fill_word_one_hots(X_nc[i], w_n, V_C, maxlen)
        X_nmask[i, 0]               = 1
        W[i]                        = 1.
        y[i, V_W.get_index(w_np1)]  = 1
    return {'w_nc': X_nc, 'w_nmask': X_nmask}, y, W
  return to_samples


def prepare_c2w2c_training_data(params, dataset, V_C, shuffle=True):
  return prepare_data(params.n_batch, dataset, to_c2w2c_samples(params, V_C), shuffle)


def prepare_c2w2w_training_data(params, dataset, V_C, V_W, shuffle=True):
  return prepare_data(params.n_batch, dataset, to_c2w2w_samples(params, V_C, V_W), shuffle)


def prepare_w2c_training_data(c2wnp1, params, dataset, V_C):
  def _to_cached_w2c_samples(cache):
    def to_samples(samples):
      cache_key = ':'.join([('' if s is None else s[0]) for s in samples])
      if cache_key in cache:
        return cache[cache_key]
      # samples not found from cache, must generate embedding first
      X, y, W = to_c2w2c_samples(params, V_C)(samples)
      W_np1e  = c2wnp1.predict({'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}, batch_size=params.n_batch)
      if len(filter(lambda s: s is None or s[1] == EOS, samples)) == len(samples):
        c2wnp1.reset_states()
      samples = ({'w_np1e': W_np1e, 'w_np1c': X['w_np1c']}, y, W)
      cache[cache_key] = samples
      return samples
    return to_samples

  return prepare_data(params.n_batch, dataset, _to_cached_w2c_samples({}), shuffle=False)
