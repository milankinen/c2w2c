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
