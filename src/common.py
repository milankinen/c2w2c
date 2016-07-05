from constants import EOW, EOS, SOS


def w2tok(word, maxlen, pad=None):
  if len(word) >= maxlen - 1:
    word = word[0: maxlen - 1]
  word += EOW
  if pad is not None and len(word) < maxlen:
    word += pad * (maxlen - len(word))
    assert len(word) == maxlen
  return word


def w2str(word):
  if word == SOS:
    return '<S>'
  elif word == EOS:
    return '</S>'
  return word
