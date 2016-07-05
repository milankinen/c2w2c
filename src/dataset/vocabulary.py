from ..constants import EOW


class Vocabulary:
  def __init__(self, tokens):
    self.tokens     = sorted(set(tokens))
    self.idx_lookup = dict((tok, idx) for idx, tok in enumerate(self.tokens))
    self.size       = len(self.tokens)

  def get_token(self, idx):
    if idx in range(0, self.size):
      return self.tokens[idx]
    raise AssertionError('Index out of bounds (%d)' % idx)

  def get_index(self, token):
    if token in self.idx_lookup:
      return self.idx_lookup[token]
    raise AssertionError('Token "%s" not found from vocabulary' % token)

  def has(self, token):
    return token in self.idx_lookup


def make_char_vocabulary(datasets):
  tokens = [tok for s in datasets for tok in s.vocabulary.tokens]
  return Vocabulary(list(''.join(tokens)) + [EOW])
