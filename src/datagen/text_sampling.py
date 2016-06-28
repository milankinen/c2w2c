import sys

import numpy as np

from training import prepare_data, to_c2w2c_samples
from ..common import w2str
from ..constants import SOS, EOW
from ..dataset import Dataset
from ..validation.helpers import word_probability_from_chars, sample_char_probabilities


def _select_best_word(w2c, w_np1e, V_C, V_W, params):
  p, w    = 0., ''
  p_chars = sample_char_probabilities(w2c, w_np1e, V_C, params)
  for tok in V_W.tokens:
    p_t = word_probability_from_chars(tok, p_chars, params.maxlen, V_C)
    if p_t > p:
      w = tok
      p = p_t
  return w


def _select_char_by_char(w2c, w_np1e, V_C, params):
  word    = ''
  p_chars = sample_char_probabilities(w2c, w_np1e, V_C, params)
  for p in p_chars:
    ch = V_C.get_token(np.argmax(p))
    if ch == EOW:
      break
    word += ch
  return word


def stdw(words):
  for w in words:
    sys.stdout.write(w2str(w) + ' ')
  sys.stdout.flush()


def _sample_step(c2wnp1, words, V_C, params):
  def to_samples(samples):
    X, _, _ = to_c2w2c_samples(params, V_C)(samples)
    return {'w_nc': X['w_nc'], 'w_nmask': X['w_nmask']}

  n_samples, _, gen = prepare_data(params.n_batch, Dataset([words + [SOS]] * params.n_batch), to_samples, shuffle=False)
  return c2wnp1.predict_generator(gen, n_samples)[-1]


def sample_c2w2c_text(c2wnp1, wc2, seed, how_many, V_W, V_C, params):
  # give seed as context before we start the actual text generation
  c2wnp1.reset_states()
  w_np1e = _sample_step(c2wnp1, seed, V_C, params)
  #stdw(seed + ['|>'])
  # then predict words word by word and sample the received word embedding
  # to actual word by using some strategy (e.g. char-by-char)
  for _ in range(0, how_many):
    #next_word  = _select_best_word(wc2, w_np1e, V_C, V_W, params)
    next_word = _select_char_by_char(wc2, w_np1e, V_C, params)
    stdw([next_word])
    w_np1e = _sample_step(c2wnp1, [next_word], V_C, params)
  print '\n'


def sample_c2w2w_text(c2w2w, seed, how_many, V_W, V_C, params):
  c2w2w.reset_states()
  P = _sample_step(c2w2w, seed, V_C, params)
  #stdw(seed + ['|>'])
  for _ in range(0, how_many):
    next_word = V_W.get_token(np.argmax(P))
    stdw([next_word])
    P = _sample_step(c2w2w, [next_word], V_C, params)
  print '\n'
