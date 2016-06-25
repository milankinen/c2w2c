from keras.callbacks import Callback


class ContextReset(Callback):
  def __init__(self, model, meta):
    super(ContextReset, self).__init__()
    self.model = model
    self.reset_points = set(reduce(lambda p, n: p + [p[-1] + n], meta, [0]))

  def on_batch_end(self, batch, logs={}):
    if (batch + 1) in self.reset_points:
      #print 'RESET', batch
      self.model.reset_states()
