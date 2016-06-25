import sys

import numpy as np

from helpers import calc_p_words_for_vocabulary, calc_p_word_for_single_word, word_probability_from_chars, sample_char_probabilities
from ..common import is_oov
from ..datagen import prepare_data, to_c2w2c_samples


def _calc_word_loss_over_vocabulary(w2c, w_np1e, expected, V_C, V_W, params):
  p_words = calc_p_words_for_vocabulary(w2c, w_np1e, V_C, V_W, params)
  p_expected = None
  for word, p_word in p_words:
    if word == expected:
      p_expected = word, p_word
      break
  if p_expected is None:
    p_expected = calc_p_word_for_single_word(w2c, w_np1e, expected, V_C, params)
    p_words.append(p_expected)

  return -np.log(p_expected[1] / np.sum(p for _, p in p_words))


def _calc_quick_loss(P_chars, expectations, V_C, maxlen):
  l, o, t = 0., 0, 0
  for idx, sample in enumerate(zip(P_chars, expectations)):
    p_chars, expected = sample
    if is_oov(expected, maxlen):
      o += 1
      continue
    word_loss = -np.log(word_probability_from_chars(expected, p_chars, maxlen, V_C))
    #print expected, '=', word_loss
    if np.isinf(word_loss):
      print 'WARN: unable to get loss of word: ' + expected
      o += 1
      continue
    l += word_loss
    t += 1
  return l, o, t


def _calc_loss(w2c, W_np1e, expectations, V_C, V_W, params):
  l, o, t = 0., 0, 0
  for idx, sample in enumerate(zip(W_np1e, expectations)):
    w_np1e, expected = sample
    if is_oov(expected, params.maxlen):
      o += 1
      continue
    # ATTENTION: this is **very expensive** operation...
    word_loss = _calc_word_loss_over_vocabulary(w2c, w_np1e, expected, V_C, V_W, params)
    #print expected, '=', word_loss
    if np.isinf(word_loss):
      print 'WARN: unable to get loss of word: ' + expected
      o += 1
      continue
    l += word_loss
    t += 1
  return l, o, t


def _calc_normalized_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, params):
  l, o, t, n = 0., 0, 0, 0
  lm.reset_states()
  while n < n_samples:
    X, expectations   = generator.next()
    W_np1e            = lm.predict(X, batch_size=n_batch)
    loss, oov, tested = _calc_loss(w2c, W_np1e, expectations, V_C, V_W, params)
    l += loss
    o += oov
    t += tested
    n += len(expectations)

  assert n == n_samples
  pp    = sys.float_info.max if t == 0 else np.exp(l / t)
  oovr  = 0 if t + o == 0 else o / float(t + o)
  return pp, oovr


def _calc_sampled_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, params):
  l, o, t, n = 0., 0, 0, 0
  lm.reset_states()
  while n < n_samples:
    X, expectations = generator.next()
    W_np1e          = lm.predict(X, batch_size=n_batch)
    for w_np1e, expected in zip(W_np1e, expectations):
      if is_oov(expected, params.maxlen):
        o += 1
        continue
      p_words, p_expected = [], None
      p_chars = sample_char_probabilities(w2c, w_np1e, V_C, params)
      for tok in V_W.tokens:
        p = word_probability_from_chars(tok, p_chars, params.maxlen, V_C)
        p_words.append(p)
        if tok == expected:
          p_expected = p
      if p_expected is None:
        p_words.append(word_probability_from_chars(expected, p_chars, params.maxlen, V_C))
        p_expected = p_words[-1]
      word_loss = -np.log(p_expected / np.sum(p_words))
      #print expected, '=', word_loss
      if np.isinf(word_loss):
        print 'WARN: unable to get loss of word: ' + expected
        o += 1
        continue
      l += word_loss
      t += 1
      n += 1
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
    X, _, _ = to_c2w2c_samples(params, V_C)(samples)
    X       = {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}
    W_np1   = list([s[1] for s in samples])
    return X, W_np1

  n_batch              = params.n_batch
  n_samples, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_normalized_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, params)


def _make_quick_test_fn(c2w2c, params, dataset, V_C):
  def to_test_samples(samples):
    X, _, _  = to_c2w2c_samples(params, V_C)(samples)
    expected = list([s[1] for s in samples])
    return X, expected

  n_batch              = params.n_batch
  n_samples, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_quick_pp(c2w2c, n_samples, n_batch, generator, V_C, params.maxlen)


def _make_sampling_test_fn(lm, w2c, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = to_c2w2c_samples(params, V_C)(samples)
    X       = {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}
    W_np1   = list([s[1] for s in samples])
    return X, W_np1

  n_batch              = params.n_batch
  n_samples, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_sampled_pp(lm, w2c, n_samples, n_batch, generator, V_C, V_W, params)


def make_c2w2c_test_function(c2w2c, lm, w2c, params, dataset, V_C, V_W):
  mode = params.validation_mode
  while 1:
    if mode == 'quick':
      return _make_quick_test_fn(c2w2c, params, dataset, V_C)
    elif mode == 'full':
      return _make_full_test_fn(lm, w2c, params, dataset, V_C, V_W)
    elif mode == 'sampled':
      return _make_sampling_test_fn(lm, w2c, params, dataset, V_C, V_W)
    else:
      print 'Invalid validation mode "%s". Using "quick" by default...' % mode
      mode = 'quick'
