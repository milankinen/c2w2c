from common import make_sentences
from constants import EOS, SOS
from vocabulary import Vocab


class TestData:
  def __init__(self, tokenized_input_str):
    sentences = make_sentences(tokenized_input_str)
    self.sentences  = sentences
    self.V_W        = Vocab([w for s in sentences for w in s] + [SOS, EOS])
    self.V_C        = Vocab(list(''.join(self.V_W.tokens)))

  def print_stats(self):
    print 'Test data statistics:'
    print '  - Number of sentences:   %d' % len(self.sentences)
    print '  - Total words:           %d' % sum(len(s) for s in self.sentences)
    print '  - Distinct words:        %d' % self.V_W.size
    print '  - Max word length:       %d' % self.V_W.maxlen
    print '  - Characters:            %d' % self.V_C.size


def load_test_data(filename):
  lines = open(filename).readlines()
  data = []
  for line in lines:
    l = line.decode('utf-8').strip('\n').strip(' ').lower()
    if len(l) > 0:
      data.append(l)
  return TestData(' \n '.join(data))
