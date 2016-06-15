import argparse, sys


class ModelParams:
  def __init__(self, args):
    self.training_dataset = args.training or 'data/training.txt'
    self.test_dataset     = args.test or 'data/test.txt'
    self.n_context        = args.context_size
    self.n_batch          = args.batch_size
    self.n_epoch          = args.num_epoch
    self.init_weight_file = args.load_weights
    self.save_weight_file = args.save_weights
    self.d_C              = args.d_C
    self.d_W              = args.d_W
    self.d_Wi             = args.d_Wi
    self.d_D              = args.d_D
    self.d_L              = args.d_L
    self.learning_rate    = args.learning_rate
    self.maxlen           = args.max_word_length
    self.limits           = args.data_limit
    self.train_data_limit = self.limits[0] if self.limits is not None else None
    self.test_data_limit  = self.limits[1] if self.limits is not None else None
    self.gen_n_samples    = args.gen_text

  def print_params(self):
    print 'Model parameters:'
    print ' - Training dataset: %s' % self.training_dataset
    print ' - Test dataset:     %s' % self.test_dataset
    if self.limits:
      print ' - Data limits:      %d / %d' % (self.train_data_limit, self.test_data_limit)
    print ' - Context size:     %d' % self.n_context
    print ' - Batch size:       %d' % self.n_batch
    print ' - Number of epoch:  %d' % self.n_epoch
    print ' - d_C:              %d' % self.d_C
    print ' - d_W:              %d' % self.d_W
    print ' - d_Wi:             %d' % self.d_Wi
    print ' - d_L:              %d' % self.d_L
    print ' - d_D:              %d' % self.d_D
    print ' - Learning rate:    %f' % self.learning_rate
    print ' - Max word length:  %d' % self.maxlen
    print ' - Load weights:     %s' % ('yes' if self.init_weight_file else 'no')
    print ' - Save weights:     %s' % ('yes' if self.save_weight_file else 'no')
    print ' - Generate samples: %s' % ('no' if self.gen_n_samples is None else str(self.gen_n_samples))


def _DataLimit(v):
  limits = v.split(":")
  if len(limits) != 2 or not limits[0].isdigit() or not limits[1].isdigit():
    raise argparse.ArgumentTypeError("Data limit must be form <training-limit>:<test-limit>, e.g. 10:1")
  return int(limits[0]), int(limits[1])

def from_cli_args():
  parser = argparse.ArgumentParser(description='C2W + W2C decoder')
  parser.add_argument('--training', metavar='filename', help='Training dataset filename')
  parser.add_argument('--test', metavar='filename', help='Validation dataset filename')
  parser.add_argument('--data-limit', '-l', metavar='training:validation', type=_DataLimit, help='Limit data size to the given rows (e.g. "10:1")')
  parser.add_argument('--validation-data-limit', metavar='n', type=int, help='Limit validation data size to the given rows')
  parser.add_argument('--context-size', metavar='n', type=int, help='Sliding window context size in training')
  parser.add_argument('--batch-size', metavar='n', type=int, help='Number of samples is single training batch')
  parser.add_argument('--learning-rate', '-r', metavar='num', type=float)
  parser.add_argument('--num-epoch', '-e', metavar='n', type=int, help='Number of epoch to run')
  parser.add_argument('--load-weights', metavar='filename', help='File containing the initial model weights')
  parser.add_argument('--save-weights', metavar='filename', help='Filename where model weights will be saved')
  parser.add_argument('--max-word-length', '-w', metavar='n', type=int, help='Maximum word length (longer words will be truncated)')
  parser.add_argument('--d_C', type=int, metavar='n', help='Character features vector size')
  parser.add_argument('--d_W', type=int, metavar='n', help='Word features vector size')
  parser.add_argument('--d_Wi', type=int, metavar='n', help='Intermediate word LSTM state dimension')
  parser.add_argument('--d_L', type=int, metavar='n', help='Language model state dimension')
  parser.add_argument('--d_D', type=int, metavar='n', help='W2C Decoder state dimension')
  parser.add_argument('--gen-text', type=int, metavar='n', help='Generate N sample sentences after each epoch')
  parser.set_defaults(context_size=10,
                      batch_size=50,
                      learning_rate=0.001,
                      num_epoch=1000,
                      max_word_length=25,
                      d_C=50,
                      d_W=300,
                      d_Wi=150,
                      d_L=1024,
                      d_D=256)

  return ModelParams(parser.parse_args())

