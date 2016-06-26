import threading
import sys
from Queue import Queue

import numpy as np

from ..common import w2tok
from ..constants import EOW, SOW, EOS
from ..dataset import Vocabulary
from ..dataset.helpers import fill_context_one_hots, fill_word_one_hots


class WordProbGen(threading.Thread):
  def __init__(self, w2c, w_np1e, V_W, V_C, params):
    super(WordProbGen,self).__init__()
    self.setDaemon(True)
    self.w2c  = w2c
    self.V_W  = V_W
    self.V_C  = V_C
    self.q    = Queue(10)
    self.p    = params
    self.we   = w_np1e

  def run(self):
    n, toks = 0, list(self.V_W.tokens)
    maxlen  = self.p.maxlen
    n_batch = self.p.n_batch
    while n < len(toks):
      n_actual  = min(len(toks) - n, n_batch)
      batch     = toks[n: n + n_batch]
      W_np1c    = np.zeros(shape=(n_actual, maxlen, self.V_C.size), dtype=np.bool)
      W_np1e    = np.repeat([self.we], n_actual, axis=0)
      fill_context_one_hots(W_np1c, batch, self.V_C, maxlen, pad=EOW)
      P_chars = self.w2c.predict({'w_np1e': W_np1e, 'w_np1c': W_np1c}, batch_size=n_actual)
      self.q.put((n_actual, zip(batch, P_chars)))
      n += n_actual


def word_probability_from_chars(word, p_chars, maxlen, V_C):
  def char_p(ch, i):
    return p_chars[i, V_C.get_index(ch)] / np.sum(p_chars[i])
  tok = w2tok(word, maxlen, pad=None)
  return np.exp(np.sum(np.log([char_p(ch, i) for i, ch in enumerate(tok)])))


def calc_p_words_for_vocabulary(w2c, w_np1e, V_C, V_W, params):
  wpgen      = WordProbGen(w2c, w_np1e, V_W, V_C, params)
  p_words, n = [], 0

  wpgen.start()
  while n < V_W.size:
    n_batch, batch = wpgen.q.get()
    for word, p_chars in batch:
      p_w = word_probability_from_chars(word, p_chars, params.maxlen, V_C)
      p_words.append((word, p_w))
    n += n_batch

  return p_words


def calc_p_word_for_single_word(w2c, w_np1e, word, V_C, params):
  p_words = calc_p_words_for_vocabulary(w2c, w_np1e, Vocabulary([word]), V_C, params)
  return p_words[0]


def sample_char_probabilities(w2c, w_np1e, V_C, params):
  maxlen  = params.maxlen
  W_np1e  = np.reshape(w_np1e, (1,) + w_np1e.shape)
  W_np1c  = np.zeros(shape=(1, maxlen, V_C.size), dtype=np.bool)
  p_chars = np.zeros(shape=(maxlen, V_C.size), dtype=np.float32)
  fill_word_one_hots(W_np1c[0], SOW, V_C, maxlen, pad=EOW)
  for i in range(0, maxlen):
    p = w2c.predict({'w_np1e': W_np1e, 'w_np1c': W_np1c}, batch_size=1)[0, i]
    np.copyto(p_chars[i], p)
    if i < maxlen - 1:
      W_np1c[0, i + 1, V_C.get_index(EOW)] = 0
      W_np1c[0, i + 1, np.argmax(p)] = 1

  return p_chars


def calc_pp(model, n_batch, meta, generator, loss_fn):
  lot = [(0., 0, 0)] * n_batch

  def _lot(samples, loss_fn):
    assert len(samples) == len(lot)
    res = []
    for idx, sample in enumerate(samples):
      predicted, expected = sample
      if expected is None:
        continue
      pl, po, pt = lot[idx]
      loss = loss_fn(predicted, expected)
      nl, no, nt = (loss, 0, 1) if (loss is not None and not np.isinf(loss)) else (0., 1, 0)
      lot[idx] = (pl + nl, po + no, pt + nt)
      if loss is not None:
        #print expected, '=', loss
        if np.isinf(loss):
          print 'WARN: unable to get loss of "%s"' % expected
      if expected == EOS:
        l, o, t = lot[idx]
        pp = sys.float_info.max if t == 0 else np.exp(l / t)
        res.append((pp, o, t))
        lot[idx] = (0., 0, 0)
    return res

  res = []
  for n in meta:
    model.reset_states()
    for _ in range(0, n):
      X, expectations   = generator.next()
      predictions       = model.predict(X, batch_size=n_batch)
      res += _lot(zip(predictions, expectations), loss_fn)

  pp    = np.mean(list(r[0] for r in res))
  t, o  = sum(r[1] for r in res), sum(r[2] for r in res)
  oovr  = 0 if t + o == 0 else o / float(t + o)
  return pp, oovr
