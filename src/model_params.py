import argparse

from util import info


class ModelParams:
  def __init__(self, args):
    self.training_dataset = args.training or 'data/training.txt'
    self.test_dataset     = args.test or 'data/test.txt'
    self.batch_size       = args.batch_size
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
    self.test_only        = args.test_only
    self.mode             = args.mode.upper()

  def print_params(self):
    info('Model hyper-parameters:')
    info(' - Mode:             %s' % self.mode)
    info(' - Training dataset: %s' % self.training_dataset)
    info(' - Test dataset:     %s' % self.test_dataset)
    if self.limits:
      info(' - Data limits:      %d / %d' % (self.train_data_limit, self.test_data_limit))
    info(' - Batch size:       %d' % self.batch_size)
    info(' - Number of epoch:  %d' % self.n_epoch)
    info(' - d_C:              %d' % self.d_C)
    info(' - d_W:              %d' % self.d_W)
    info(' - d_Wi:             %d' % self.d_Wi)
    info(' - d_L:              %d' % self.d_L)
    info(' - d_D:              %d' % self.d_D)
    info(' - Learning rate:    %f' % self.learning_rate)
    info(' - Max word length:  %d' % self.maxlen)
    info(' - Load weights:     %s' % ('yes' if self.init_weight_file else 'no'))
    info(' - Save weights:     %s' % ('yes' if self.save_weight_file else 'no'))
    info(' - Generate samples: %s' % ('no' if self.gen_n_samples is None else str(self.gen_n_samples)))
    info(' - Run only tests:   %s' % ('yes' if self.test_only else 'no'))


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
  parser.add_argument('--test-only', '-T', action='store_true', help='Run only PP test and (optional) text generation')
  parser.add_argument('--mode', metavar='c2w2c|word', help='Select which mode to run')
  parser.set_defaults(batch_size=50,
                      learning_rate=0.001,
                      num_epoch=1000,
                      max_word_length=25,
                      test_only=False,
                      mode='c2w2c',
                      d_C=50,
                      d_W=50,
                      d_Wi=150,
                      d_L=500,
                      d_D=500)

  return ModelParams(parser.parse_args())

