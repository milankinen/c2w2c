from constants import EOW, EOS, SOS


def make_sentences(tokenized_input_str):
  tokens = tokenized_input_str.lower().split(' ')
  sentences = [[SOS]]
  for tok in tokens:
    sentences[-1].append(tok + EOW)
    if tok in ['.', '!', '?']:
      sentences[-1].append(EOS)
      sentences.append([SOS])
  # remove last placeholder sentence
  sentences.pop()
  return sentences


def pad(word, V_W):
  return word + (EOW * (V_W.maxlen - len(word)))


def fill_char_indices(X, word, V_W, V_C):
  padded = pad(word, V_W)
  for i, ch in enumerate(padded):
    X[i] = V_C.get_index(ch)


def fill_context_indices(X, ctx, V_W, V_C):
  n = len(ctx)
  for i in range(0, n):
    fill_char_indices(X[i], ctx[i], V_W, V_C)


def fill_char_one_hots(X, word, V_W, V_C):
  padded = pad(word, V_W)
  for i, ch in enumerate(padded):
    X[i, V_C.get_index(ch)] = 1
