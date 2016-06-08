import numpy as np

from vocabulary import Vocab

# end of word character
EOW = '\n'

# start of sentence token
SOS = '<S>'
# end of sentence token
EOS = '</S>'


class TrainingData:
  def __init__(self, tokenized_input_str):
    tokens = tokenized_input_str.lower().split(' ')
    sentences = [[SOS]]
    for tok in tokens:
      sentences[-1].append((tok + EOW))
      if tok in ['.', '!', '?']:
        sentences[-1].append(EOS)
        sentences.append([SOS])
    # remove last placeholder sentence
    sentences.pop()
    self.words  = [w for s in sentences for w in s]
    self.V_W    = Vocab(tokens + [SOS, EOS])
    self.V_C    = Vocab(list(''.join(self.V_W.tokens)) + [EOW])

  def print_stats(self):
    print 'Training data statistics:'
    print '  - Words:    %d' % len(self.words)
    print '  - Distinct: %d' % len(self.V_W.dim[0])
    print '  - Chars:    %d' % len(self.V_C.dim[0])

  def get_num_samples(self, n_context):
    return len(self.words) - n_context - 1

  def as_generator(self, n_context, batch_size):
    V_W       = self.V_W
    V_C       = self.V_C
    words     = self.words
    num_words = len(words)

    def generator():
      idx = 0
      while 1:
        actual_size = min(batch_size, num_words - idx - n_context - 1)
        X = np.zeros(shape=(actual_size, n_context, V_W.dim[1]), dtype=np.int32)
        y = np.zeros(shape=(actual_size, V_W.dim[1], V_C.dim[0]), dtype=np.bool)
        for i in range(0, actual_size):
          ctx = words[idx:idx + n_context]
          to_predict = words[idx + n_context]
          for j in range(0, n_context):
            word = ctx[j]
            for k in range(0, V_W.dim[1]):
              X[i, j, k] = V_C.get_index(word[k] if len(word) > k else EOW)
          for k in range(0, V_W.dim[1]):
            y[i, k, V_C.get_index(to_predict[k] if len(to_predict) > k else EOW)] = 1
        idx += actual_size
        if idx <= num_words:
          idx = 0
        yield (X, y)

    return generator()


def load_training_data(filename):
  lines = open(filename).readlines()
  data = []
  for line in lines:
    l = line.decode('utf-8').strip('\n').strip(' ').lower()
    if len(l) > 0:
      data.append(l)
  return TrainingData(' '.join(data))
