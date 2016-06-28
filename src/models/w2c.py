from keras.layers import Input, Activation, RepeatVector
from keras.models import Model

from ..layers import ProjectionOverTime, DecoderGRU, Maxout


class W2C(Model):
  def __init__(self, maxlen, d_W, d_C, d_D, V_C, trainable=True, apply_softmax=False):
    """
      n_batch  :: batch size for model application
      maxlen   :: maximum sampled word length
      d_W      :: word features
      d_C      :: character features
      d_D      :: internal decoder state dimension
      V_C      :: character vocabulary
    """

    w_np1E  = Input(shape=(d_W, ), name='w_np1e', dtype='floatX')
    w_np1c  = Input(shape=(maxlen, V_C.size), name='w_np1c', dtype='int8')

    w_np1ce = ProjectionOverTime(d_C, trainable=trainable)(w_np1c)
    h       = DecoderGRU(d_D, trainable=trainable, return_sequences=True)([w_np1ce, w_np1E])
    s       = Maxout(d_D, trainable=trainable)([h, w_np1ce, RepeatVector(maxlen)(w_np1E)])
    c_I     = ProjectionOverTime(V_C.size, trainable=trainable)(s)

    # for W2C training only
    if apply_softmax:
      c_I = Activation('softmax')(c_I)

    super(W2C, self).__init__(input=[w_np1E, w_np1c], output=c_I, name='W2C')
