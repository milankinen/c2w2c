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


def _sample_words(model, c, maxlen, V_C, K=20):
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
  best_chars, losses = beamsearch(predict, eow, k=K, maxsample=maxlen)
  best_words = []
  for word_chars in best_chars:
    word = ""
    for ch in word_chars:
      if ch == eow:
        break
      word += V_C.get_token(ch)
    best_words.append(word)
  probs = 1. / np.exp(np.array(losses))
  return best_words, probs


def _sample_word(model, c, maxlen, V_C, K=20):
  words, probs = _sample_words(model, c, maxlen, V_C, K)
  idx = _select(probs)
  return words[idx]


def _sample_sent(model, step, c_initial, maxlen, V_C, max_samples, K_s=5, K_w=20):
  def predict(samples):
    preds, probs = [], []
    for ctx in samples:
      model.restore_states()
      c = c_initial if len(ctx) == 0 else step(ctx)
      pe, po = _sample_words(model, c, maxlen, V_C, K=K_w)
      preds.append(pe)
      probs.append(po)
    p_all = np.array(probs)
    return p_all, preds

  sents, losses = beamsearch(predict, EOS, k=K_s, maxsample=max_samples)
  probs = 1. / np.exp(np.array(losses))
  idx = _select(probs, .15)
  return sents[idx]


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
  for w in seed:
    _stdwrite(w)
  print '|>\n'
  model.reset_states()
  n = 0
  while n < how_many:
    c = _step(seed)
    model.save_states()
    next_words = _sample_sent(model, _step, c, maxlen, V_C, max_samples=(how_many - n))
    for w in next_words:
      _stdwrite(w)
    n += len(next_words)
    seed = next_words
    model.restore_states()
