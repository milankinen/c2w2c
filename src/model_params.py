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
    self.learning_rate    = args.learning_rate
    self.maxlen           = args.max_word_length

  def print_params(self):
    print 'Model parameters:'
    print ' - Training dataset: %s' % self.training_dataset
    print ' - Test dataset:     %s' % self.test_dataset
    print ' - Context size:     %d' % self.n_context
    print ' - Batch size:       %d' % self.n_batch
    print ' - Number of epoch:  %d' % self.n_epoch
    print ' - d_C:              %d' % self.d_C
    print ' - d_W:              %d' % self.d_W
    print ' - d_Wi:             %d' % self.d_Wi
    print ' - d_D:              %d' % self.d_D
    print ' - Learning rate:    %d' % self.learning_rate
    print ' - Max word length:  %d' % self.maxlen
    print ' - Load weights:     %s' % ('yes' if self.init_weight_file else 'no')
    print ' - Save weights:     %s' % ('yes' if self.save_weight_file else 'no')


def from_cli_args():
  parser = argparse.ArgumentParser(description='C2W + W2C decoder')
  parser.add_argument('--training')
  parser.add_argument('--test')
  parser.add_argument('--context-size', type=int)
  parser.add_argument('--batch-size', type=int)
  parser.add_argument('--learning-rate', type=float)
  parser.add_argument('--num-epoch', type=int)
  parser.add_argument('--load-weights')
  parser.add_argument('--save-weights')
  parser.add_argument('--max-word-length')
  parser.add_argument('--d_C', type=int)
  parser.add_argument('--d_W', type=int)
  parser.add_argument('--d_Wi', type=int)
  parser.add_argument('--d_D', type=int)
  parser.set_defaults(context_size=5,
                      batch_size=50,
                      learning_rate=0.0001,
                      num_epoch=1000,
                      max_word_length=25,
                      d_C=50,
                      d_W=300,
                      d_Wi=150,
                      d_D=1024)

  return ModelParams(parser.parse_args())

