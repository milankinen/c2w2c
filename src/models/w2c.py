from keras.layers import LSTM, Input, RepeatVector, Activation, merge
from keras.models import Model

from ..layers import ProjectionOverTime


class W2C(Model):
  def __init__(self, n_batch, maxlen, d_W, d_D, V_C, trainable=True, apply_softmax=False):
    """
      n_batch  :: batch size for model application
      maxlen   :: maximum sampled word length
      d_L      :: language model state dimension (input embedding vector size)
      d_D      :: internal decoder state dimension
      V_C      :: character vocabulary
    """

    w_np1E  = Input(batch_shape=(n_batch, d_W), name='w_np1e', dtype='floatX')
    w_np1c  = Input(batch_shape=(n_batch, maxlen, V_C.size), name='w_np1c', dtype='int8')

    w_E     = RepeatVector(maxlen)(w_np1E)
    w_EC    = merge(inputs=[w_E, w_np1c], mode='concat')
    c_E     = LSTM(d_D,
                   trainable=trainable,
                   return_sequences=True,
                   consume_less='gpu')(w_EC)

    c_I     = ProjectionOverTime(V_C.size, trainable=trainable)(c_E)

    # for W2C training only
    if apply_softmax:
      c_I = Activation('softmax')(c_I)

    super(W2C, self).__init__(input=[w_np1E, w_np1c], output=c_I, name='W2C')
