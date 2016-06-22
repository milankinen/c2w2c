from keras.layers import LSTM, Input, Reshape
from keras.models import Model

from ..layers import LMMask


class LanguageModel(Model):
  def __init__(self, n_batch, d_W, d_L, trainable=True):
    """
      n_batch  :: batch size for model application
      d_L      :: language model state dimension (and output vector size)
      d_W      :: input word embedding size (word features)
    """

    w_n           = Input(batch_shape=(n_batch, d_W), name='w_n', dtype='floatX')
    w_nmask       = Input(batch_shape=(n_batch, 1), name='w_nmask', dtype='int8')

    # Prevent padded samples to affect internal state (and cause NaN loss in worst
    # case) by masking them by using another input value
    w_nmasked     = LMMask(0.)([Reshape((1, d_W))(w_n), w_nmask])

    # Using stateful LSTM for language model - store layer as member so that the
    # model fitting code resets the state after each sentence
    self._lstm    = LSTM(d_L,
                         trainable=trainable,
                         return_sequences=False,
                         stateful=True,
                         consume_less='gpu')

    w_np1E        = self._lstm(w_nmasked)

    super(LanguageModel, self).__init__(input=[w_n, w_nmask], output=w_np1E, name='LanguageModel')

  def reset_state(self):
    self._lstm.reset_states()

