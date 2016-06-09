from constants import EOS, SOS


def make_sentences(tokenized_input_str):
  tokens = tokenized_input_str.lower().split(' ')
  sentences = [[SOS]]
  for tok in tokens:
    tok = tok.strip(' ')
    if len(tok) > 0:
      sentences[-1].append(tok.strip(' '))
      if tok in ['.', '!', '?', '\n'] and len(sentences[-1]) > 1:
        sentences[-1].append(EOS)
        sentences.append([SOS])
  # remove last placeholder sentence
  sentences.pop()
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
