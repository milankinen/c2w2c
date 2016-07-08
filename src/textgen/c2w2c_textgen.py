import sys

import numpy as np

from search import beamsearch
from ..common import EOS, EOW, w2str
from ..dataset import initialize_c2w2c_data, Dataset


def _stdwrite(word):
  sys.stdout.write(w2str(word) + ' ')
  sys.stdout.flush()


def _select(a, temperature=.1):
  a = np.log(a) / temperature
  a = np.exp(a) / np.sum(np.exp(a))
  return np.argmax(np.random.multinomial(1, a, 1))


def _sample_word(model, c, maxlen, V_C, K=20):
  def predict(samples):
    context = np.array([c] * len(samples))
    prev_chars = np.zeros((len(samples), maxlen), dtype=np.int32)
    probs = np.zeros((len(samples), V_C.size), dtype=np.float32)
    for i, prev in enumerate(samples):
      for j, ch in enumerate(prev):
        prev_chars[i, j + 1] = ch + 1
    preds = model.predict_chars(context, prev_chars)
    for i, prev in enumerate(samples):
      np.copyto(probs[i], preds[i, len(prev)])
    return probs

  eow = V_C.get_index(EOW)
  best_chars, losses = beamsearch(predict, 0, eow, k=K, maxsample=maxlen)
  best_words = []
  for word_chars in best_chars:
    word = ""
    for ch in word_chars:
      if ch == eow:
        break
      word += V_C.get_token(ch)
    best_words.append(word)
  probs = 1. / np.exp(np.array(losses))
  idx = _select(probs)
  return best_words[idx]


def _sample_sent(model, maxlen, V_C, K_s=5, K_w=20):
  pass


def generate_c2w2c_text(model, maxlen, seed, how_many):
  hyper_params    = model.get_hyperparams()
  batch_size, V_C = hyper_params[0], hyper_params[-1]

  def _step(words):
    sents = [words + [EOS]] * batch_size
    make_gen, _ = initialize_c2w2c_data(Dataset(sents), batch_size, maxlen, V_C, shuffle=False)
    n_samples, gen = make_gen()
    ctx_pred = model.predict_ctx(gen, n_samples)
    return ctx_pred[-1]

  print ''
  model.reset_states()
  for _ in range(how_many):
    c = _step(seed)
    next_word = _sample_word(model, c, maxlen, V_C)
    _stdwrite(next_word)
    seed = [next_word]
