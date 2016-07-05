import sys

import numpy as np

from ..common import EOS, w2str
from ..dataset import initialize_word_lstm_data, Dataset


def _stdwrite(word):
  sys.stdout.write(w2str(word) + ' ')
  sys.stdout.flush()


def generate_word_lstm_text(model, seed, how_many):
  hyper_params    = model.get_hyperparams()
  batch_size, V_W = hyper_params[0], hyper_params[-1]

  def _step(words):
    sents = [words + [EOS]] * batch_size
    make_gen, _ = initialize_word_lstm_data(Dataset(sents), batch_size, V_W, shuffle=False)
    n_samples, gen = make_gen()
    p_words = model.predict_generator(gen, n_samples)
    return p_words[-1]

  print ''
  model.reset_states()
  for _ in range(how_many):
    p = _step(seed)
    next_word = V_W.get_token(np.argmax(p))
    _stdwrite(next_word)
    seed = [next_word]
