import numpy as np
import sys

from keras.callbacks import Callback

SAMPLES_IN_MINI_ITERATION_TEST = 10


class MiniIteration(Callback):
  def __init__(self, prev_pp, sentence_seq, model, run_tests, run_minitest_after=None):
    super(MiniIteration, self).__init__()
    self.model      = model
    self.weights    = np.array(self.model.get_weights())
    self.run_tests  = run_tests
    self.i          = sentence_seq[0]
    self.seq        = sentence_seq[1:]
    self.pp         = sys.float_info.max if prev_pp is None else prev_pp
    self.interval   = run_minitest_after
    self.reset      = False

  def _run_pp_minitest(self):
    pp, _ = self.run_tests(SAMPLES_IN_MINI_ITERATION_TEST)
    if pp <= self.pp:
      # better perplexity -> save current weights
      self.pp = pp
      self.weights = np.array(self.model.get_weights())
      self.reset = False
    else:
      # mark weights to be reset during next model state reset
      self.reset = True

  def _reset_model(self):
    m = self.model
    m.reset_states()
    if self.reset:
      m.set_weights(self.weights)
    self.reset = False

  def on_batch_begin(self, batch, logs={}):
    if self.i <= 0:
      assert len(self.seq) > 0
      self.i = self.seq[0]
      self.seq = self.seq[1:]

  def on_batch_end(self, batch, logs={}):
    if self.interval is not None and batch != 0 and batch % self.interval == 0:
      self._run_pp_minitest()

    self.i -= 1
    if self.i <= 0:
      self._reset_model()

