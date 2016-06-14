import theano as T

from keras import backend as K
from keras.layers import TimeDistributed


class FastDistribute(TimeDistributed):
  def __init__(self, layer, **kwargs):
    super(FastDistribute, self).__init__(layer, **kwargs)

  def call(self, X, mask=None):
    input_shape = self.input_spec[0].shape
    if input_shape[0]:
      raise AssertionError("Batch size aware RNN not supported")
    if not input_shape[1]:
      raise AssertionError("Timestep must be known")

    # flatten and skip duplicates
    X = K.concatenate([X.take(0, axis=1)[:-1], X.take(-1, axis=0)], axis=0)
    # run inner model
    Y = self.layer.call(X)

    def _sliding(*y):
      return [K.concatenate(list(K.reshape(y_i, (1, y_i.shape[0])) for y_i in y), axis=0)]

    seqs = [{'input': Y, 'taps': list(range(0, input_shape[1]))}]
    output, _ = T.scan(_sliding, sequences=seqs)
    return output
