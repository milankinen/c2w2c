import numpy as np
import sys

from ..common import w2tok, is_oov
from ..constants import SOW, EOW
from ..dataset.helpers import fill_word_one_hots
from ..datagen import _prepare_data, _to_c2w2c_samples


def _calc_word_probability(word, p_chars, maxlen, V_C):
  def char_p(ch, i):
    return p_chars[i, V_C.get_index(ch)] / np.sum(p_chars[i])
  tok = w2tok(word, maxlen, pad=EOW)
  return np.prod([char_p(ch, i) for i, ch in enumerate(tok)])


def _calc_word_loss_over_vocabulary(w2c, w_np1e, expected, V_C, V_W, maxlen):
  W_np1e = np.reshape(w_np1e, (1,) + w_np1e.shape)
  p_words, p_expected = [], None

  def p_chars(word):
    w_np1c = np.zeros(shape=(1, maxlen, V_C.size), dtype=np.bool)
    fill_word_one_hots(w_np1c[0], word, V_C, maxlen, pad=EOW)
    return w2c.predict({'w_np1e': W_np1e, 'w_np1c': w_np1c}, batch_size=1)[0]

  for w in V_W.tokens:
    p_w = _calc_word_probability(w, p_chars(w), maxlen, V_C)
    if w == expected:
      p_expected = p_w
    p_words.append(p_w)
  assert p_expected is not None
  #print 'p_expected', p_expected
  return -np.log(p_expected / np.sum(p_words))


def _calc_quick_loss(P_chars, expectations, V_C, maxlen):
  l, o, t = 0., 0, 0
  for idx, sample in enumerate(zip(P_chars, expectations)):
    p_chars, expected = sample
    if is_oov(expected, maxlen):
      o += 1
      continue
    word_loss = -np.log(_calc_word_probability(expected, p_chars, maxlen, V_C))
    if np.isnan(word_loss):
      print 'WARN: unable to get loss of word: ' + expected
      o += 1
      continue
    l += word_loss
    t += 1
  #print 'loss', l
  return l, o, t


def _calc_loss(w2c, W_np1e, expectations, V_C, V_W, maxlen):
  l, o, t = 0., 0, 0
  for idx, sample in enumerate(zip(W_np1e, expectations)):
    w_np1e, expected = sample
    if is_oov(expected, maxlen):
      o += 1
      continue
    # ATTENTION: this is **very expensive** operation...
    word_loss = _calc_word_loss_over_vocabulary(w2c, w_np1e, expected, V_C, V_W, maxlen)
    if np.isnan(word_loss):
      print 'WARN: unable to get loss of word: ' + expected
      o += 1
      continue
    l += word_loss
    t += 1
  #print 'loss', l
  return l, o, t


def _calc_normalized_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, maxlen):
  l, o, t, n = 0., 0, 0, 0
  lm.reset_states()
  while n < n_samples:
    X, expectations   = generator.next()
    W_np1e            = lm.predict(X, batch_size=n_batch)
    loss, oov, tested = _calc_loss(w2c, W_np1e, expectations, V_C, V_W, maxlen)
    l += loss
    o += oov
    t += tested
    n += len(expectations)

  assert n == n_samples
  pp    = sys.float_info.max if t == 0 else np.exp(l / t)
  oovr  = 0 if t + o == 0 else o / float(t + o)
  return pp, oovr


def _calc_quick_pp(c2w2c, n_samples, n_batch, generator, V_C, maxlen):
  l, o, t, n = 0., 0, 0, 0
  c2w2c.reset_states()
  while n < n_samples:
    X, expectations   = generator.next()
    predictions       = c2w2c.predict(X, batch_size=n_batch)
    loss, oov, tested = _calc_quick_loss(predictions, expectations, V_C, maxlen)
    l += loss
    o += oov
    t += tested
    n += len(expectations)

  assert n == n_samples
  pp    = sys.float_info.max if t == 0 else np.exp(l / t)
  oovr  = 0 if t + o == 0 else o / float(t + o)
  return pp, oovr


def _make_full_test_fn(lm, w2c, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = _to_c2w2c_samples(params, V_C)(samples)
    X       = {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}
    W_np1   = list([s[1] for s in samples])
    return X, W_np1

  n_batch              = params.n_batch
  n_samples, generator = _prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return (lamda : _calc_normalized_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, params.maxlen))


def _make_quick_test_fn(c2w2c, params, dataset, V_C):
  def to_test_samples(samples):
    X, _, _  = _to_c2w2c_samples(params, V_C)(samples)
    expected = list([s[1] for s in samples])
    return X, expected

  n_batch              = params.n_batch
  n_samples, generator = _prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return (lamda : _calc_quick_pp(c2w2c, n_samples, n_batch, generator, V_C, params.maxlen))


def make_c2w2c_test_function(c2w2c, lm, w2c, params, dataset, V_C, V_W):
  QUICK_MODE = True
  if QUICK_MODE:
    return _make_quick_test_fn(c2w2c, params, dataset, V_C)
  else:
    return _make_full_test_fn(lm, w2c, params, dataset, V_C, V_W)
