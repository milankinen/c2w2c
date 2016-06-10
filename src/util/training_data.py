import random

import numpy as np

from constants import EOS, SOS
from common import make_sentences, fill_context_indices, fill_char_one_hots, load_input
from vocabulary import Vocab


class TrainingData:
  def __init__(self, tokenized_lines, test_data):
    sentences = make_sentences(tokenized_lines)
    self.sentences  = sentences
    self.n_words    = sum(len(s) for s in self.sentences)
    self.V_Wm       = Vocab([w for s in sentences for w in s] + [SOS, EOS])
    # ensure that we don't need to discard characters or clip words from the test set
    self.V_W        = Vocab(self.V_Wm.tokens + test_data.V_W.tokens)
    self.V_C        = Vocab(test_data.V_C.tokens + list(''.join(self.V_W.tokens)))

  def print_stats(self):
    print 'Training data statistics:'
    print '  - Number of sentences:   %d' % len(self.sentences)
    print '  - Total words:           %d' % self.n_words
    print '  - Distinct words:        %d' % self.V_Wm.size
    print '  - Max word length:       %d' % self.V_Wm.maxlen
    print '  - Characters:            %d' % self.V_C.size
    print '  - Sample dimension:      (ctx, %d, %d)' % (self.V_W.maxlen, self.V_C.size)

  def get_num_samples(self, n_context):
    return self.n_words - n_context - 1

  def as_generator(self, n_context, batch_size):
    V_W     = self.V_W
    V_C     = self.V_C
    n_words = self.n_words
    sents   = self.sentences[:]
    idx     = 0

    random.shuffle(sents)
    words = [w for s in sents for w in s]
    while 1:
      actual_size = min(batch_size, n_words - idx - n_context - 1)
      X = np.zeros(shape=(actual_size, n_context, V_W.maxlen), dtype=np.int32)
      y = np.zeros(shape=(actual_size, V_W.maxlen, V_C.size), dtype=np.bool)
      for i in range(0, actual_size):
        fill_context_indices(X[i], words[idx + i:idx + i + n_context], V_W, V_C)
        fill_char_one_hots(y[i], words[idx + i + n_context], V_W, V_C)
      idx += actual_size
      if idx >= n_words:
        idx = 0
      yield (X, y)

  def make_test_sentences(self, test_data):
    sents = test_data.sentences
    V_C   = self.V_C
    V_W   = self.V_W
    X     = []
    for s in sents:
      x = np.zeros(shape=(1, len(s) - 1, V_W.maxlen), dtype=np.int32)
      fill_context_indices(x[0], s[:-1], V_W, V_C)
      X.append((s[1:], x))
    return X


def load_training_data(filename, test_data):
  return TrainingData(load_input(filename), test_data)
