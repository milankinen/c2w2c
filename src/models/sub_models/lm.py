import numpy as np

from keras import backend as K
from keras.layers import LSTM, Lambda, Input, Reshape
from keras.models import Model

# define how many LSTM layers will be used for language model
NUM_LSTMs = 2


class LanguageModel(Model):
  def __init__(self, batch_size, d_W, d_L):
    """
      batch_size = batch size used in training/validation (mandatory because of stateful LSTMs)
      n_ctx      = context size in training/validation
      d_W        = word features (of output word embeddings from C2W sub-model)
      d_L        = language model hidden state size
    """
    def masked_ctx(emb, mask):
      class L(Lambda):
        def __init__(self):
          super(L, self).__init__(lambda x: x[0] * K.expand_dims(x[1], -1), lambda input_shapes: input_shapes[0])

        def compute_mask(self, x, input_mask=None):
          return K.expand_dims(x[1], -1)
      return L()([Reshape((1, d_W))(emb), mask])

    self._saved_states = None
    self._lstms = []

    ctx_emb   = Input(batch_shape=(batch_size, d_W), name='ctx_emb')
    ctx_mask  = Input(batch_shape=(batch_size,), name='ctx_mask')

    C = masked_ctx(ctx_emb, ctx_mask)
    for i in range(NUM_LSTMs):
      lstm = LSTM(d_L,
                  return_sequences=(i < NUM_LSTMs - 1),
                  stateful=True,
                  consume_less='gpu')
      self._lstms.append(lstm)
      C = lstm(C)

    super(LanguageModel, self).__init__(input=[ctx_emb, ctx_mask], output=C, name='LanguageModel')

  def save_states(self):
    states = []
    for lstm in self._lstms:
      states.append([np.copy(K.get_value(s)) for s in lstm.states])
    self._saved_states = states

  def restore_states(self):
    assert self._saved_states is not None
    for states, lstm in zip(self._saved_states, self._lstms):
      for src, dest in zip(states, lstm.states):
        K.set_value(dest, src)

  def reset_states(self):
    self._saved_states = None
    return super(LanguageModel, self).reset_states()


"""
from keras.engine import Input
import numpy as np

m = LanguageModel(2, 2, 3, 4)
m.compile(optimizer='sgd', loss='mse')

Xe = np.array([[[.1, .2, .3],
                [.1, .2, .3]],
               [[.2, .3, .3],
                [.2, .3, .3]]])

Xm = np.array([[1., 0.],
               [0., 0.]])

res = m.predict({'ctx_emb': Xe, 'ctx_mask': Xm}, batch_size=2).tolist()
assert res[0][0] == res[0][1]
assert not np.any(res[1])
raise "Done"
"""
