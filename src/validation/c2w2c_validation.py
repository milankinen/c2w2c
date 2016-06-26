import numpy as np

from helpers import calc_p_words_for_vocabulary, calc_p_word_for_single_word, word_probability_from_chars, sample_char_probabilities, calc_pp
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


def _calc_normalized_pp(lm, w2c, meta, n_batch, generator, V_C, V_W, params):
  def loss_fn(w_np1e, expected):
    if not is_oov(expected, params.maxlen):
      word_loss = _calc_word_loss_over_vocabulary(w2c, w_np1e, expected, V_C, V_W, params)
      return word_loss

  return calc_pp(lm, n_batch, meta, generator, loss_fn)


def _calc_sampled_pp(lm, w2c, meta, n_batch, generator, V_C, V_W, params):
  def loss_fn(w_np1e, expected):
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
    return word_loss

  return calc_pp(lm, n_batch, meta, generator, loss_fn)


def _calc_quick_pp(c2w2c, meta, n_batch, generator, V_C, maxlen):
  def loss_fn(p_chars, expected):
    if not is_oov(expected, maxlen):
      word_loss = -np.log(word_probability_from_chars(expected, p_chars, maxlen, V_C))
      return word_loss

  return calc_pp(c2w2c, n_batch, meta, generator, loss_fn)


def _make_full_test_fn(lm, w2c, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = to_c2w2c_samples(params, V_C)(samples)
    X       = {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}
    W_np1   = list([(None if s is None else s[1]) for s in samples])
    return X, W_np1

  n_batch                    = params.n_batch
  n_samples, meta, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_normalized_pp(lm, w2c, meta, n_batch, generator, V_C, V_W, params)


def _make_quick_test_fn(c2w2c, params, dataset, V_C):
  def to_test_samples(samples):
    X, _, _  = to_c2w2c_samples(params, V_C)(samples)
    expected = list([(None if s is None else s[1]) for s in samples])
    return X, expected

  n_batch                    = params.n_batch
  n_samples, meta, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_quick_pp(c2w2c, meta, n_batch, generator, V_C, params.maxlen)


def _make_sampling_test_fn(lm, w2c, params, dataset, V_C, V_W):
  def to_test_samples(samples):
    X, _, _ = to_c2w2c_samples(params, V_C)(samples)
    X       = {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}
    W_np1   = list([(None if s is None else s[1]) for s in samples])
    return X, W_np1

  n_batch                    = params.n_batch
  n_samples, meta, generator = prepare_data(n_batch, dataset, to_test_samples, shuffle=False)
  return lambda: _calc_sampled_pp(lm, w2c, meta, n_batch, generator, V_C, V_W, params)


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
