from constants import SOW, EOW, EOS, SOS


def is_oov(word, maxlen):
  return len(word) >= maxlen - 2


def w2tok(word, maxlen):
  if word == SOS or word == EOS:
    # SOS and EOS already have SOW and EOW
    return word
  if len(word) >= maxlen - 2:
    word = word[0: maxlen - 2]
  return SOW + word + EOW


def w2str(word):
  if word == SOS:
    return '<S>'
  elif word == EOS:
    return '</S>'
  return word
