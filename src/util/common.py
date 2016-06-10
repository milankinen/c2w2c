from constants import EOS, SOS


def load_input(filename, max=None):
  lines = open(filename).readlines()
  data = []
  for line in lines:
    l = line.decode('utf-8').strip('\n').strip(' ')
    if len(l) > 0:
      data.append(l)
  return data if max is None else data[0:max]


def make_sentences(tokenized_lines):
  sentences = []
  for line in tokenized_lines:
    tokens = line.split(' ')
    sentences.append([SOS])
    for tok in tokens:
      tok = tok.strip(' ')
      if len(tok) > 0:
        sentences[-1].append(tok)
        if tok in ['.', '!', '?'] and len(sentences[-1]) > 1:
          sentences[-1].append(EOS)
          sentences.append([SOS])
    if len(sentences[-1]) == 1:
      sentences.pop()
    else:
      sentences[-1].append(EOS)
  return sentences


def fill_char_indices(X, word, V_W, V_C):
  for i, ch in enumerate(word):
    X[i] = V_C.get_index(ch)


def fill_context_indices(X, ctx, V_W, V_C):
  n = len(ctx)
  for i in range(0, n):
    fill_char_indices(X[i], ctx[i], V_W, V_C)


def fill_char_one_hots(X, word, V_W, V_C):
  for i, ch in enumerate(word):
    X[i, V_C.get_index(ch)] = 1
