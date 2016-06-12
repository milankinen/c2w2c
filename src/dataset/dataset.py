from constants import EOS, SOS
from vocabulary import Vocabulary


def _load_input(filename, max):
  lines = open(filename).readlines()
  data = []
  for line in lines:
    l = line.decode('utf-8').strip('\n').strip(' ')
    if len(l) > 0:
      data.append(l)
  return data if max is None else data[0:max]


def _make_sentences(tokenized_lines):
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


class Dataset:
  def __init__(self, sentences):
    self.sentences  = sentences
    self.n_words    = sum(len(s) for s in self.sentences)
    self.vocabulary = Vocabulary(self.get_words())

  def print_stats(self):
    print 'Dataset statistics:'
    print '  - Number of sentences:   %d' % len(self.sentences)
    print '  - Number of words:       %d' % self.n_words
    print '  - Distinct words:        %d' % self.vocabulary.size

  def get_words(self):
    return [w for s in self.sentences for w in s]


def load_dataset(filename, max_lines=None):
  return Dataset(_make_sentences(_load_input(filename, max_lines)))


def make_char_vocabulary(datasets):
  tokens = [tok for s in datasets for tok in s.vocabulary.tokens]
  return Vocabulary(list(''.join(tokens)))
