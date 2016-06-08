class Vocab:
  def __init__(self, tokens):
    self.tokens     = sorted(set(tokens))
    self.idx_lookup = dict((tok, idx) for idx, tok in enumerate(self.tokens))
    self.dim        = (len(self.tokens), max(len(tok) for tok in self.tokens))

  def get_token(self, idx):
    if idx in range(0, self.dim[0]):
      return self.tokens[idx]
    return None

  def get_index(self, token):
    if token in self.idx_lookup:
      return self.idx_lookup[token]
    return None
