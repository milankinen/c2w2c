import sys

from keras.callbacks import Callback


class MinibatchValidator(Callback):
  def __init__(self, interval, prev_pp, sentence_seq, reset_model, run_tests, get_weights, set_weights):
    super(MinibatchValidator, self).__init__()
    self.reset_model    = reset_model
    self.run_tests      = run_tests
    self.get_weights    = get_weights
    self.set_weights    = set_weights
    self.weights        = get_weights()
    self.i              = sentence_seq[0]
    self.seq            = sentence_seq[1:]
    self.pp             = sys.float_info.max if prev_pp is None else prev_pp
    self.interval       = interval
    self.must_reset     = False

  def on_batch_begin(self, batch, logs={}):
    if self.i <= 0:
      assert len(self.seq) > 0
      self.i = self.seq[0]
      self.seq = self.seq[1:]

  def on_batch_end(self, batch, logs={}):
    if batch != 0 and batch % self.interval == 0 and False:
      pp, _ = self.run_tests(10)
      if pp <= self.pp:
        self.pp = pp
        self.weights = self.get_weights()
        self.must_reset = False
      else:
        self.must_reset = True

    self.i -= 1
    if self.i <= 0:
      self.reset_model()
      if self.must_reset:
        self.set_weights(self.weights)
        self.must_reset = False

