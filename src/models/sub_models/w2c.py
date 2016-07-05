from keras import backend as K
from keras.layers import Input, RepeatVector, Dropout, Embedding
from keras.models import Model

from ..layers import ProjectionOverTime, DecoderGRU, Maxout


class W2C(Model):
  def __init__(self, maxlen, d_L, d_C, d_D, V_C):
    """
    maxlen = maximum input/output word size
    d_L    = language model hidden state (= context vector) size
    d_C    = character features (input embedding vector size)
    d_D    = decoder hidden state h size
    V_C    = character vocabulary
    """
    # extend embeddings to treat zero values as zeros vectors (for y_0 = 0)
    # but don't do any masking
    class CharEmb(Embedding):
      def call(self, x, mask=None):
        y = super(CharEmb, self).call(x)
        return y * K.cast(K.expand_dims(x, -1), K.floatx())

    c       = Input(shape=(d_L,), name='c')
    y_tm1   = Input(shape=(maxlen,), name='y_tm1', dtype='int32')

    ye_tm1  = CharEmb(V_C.size + 1, d_C)(y_tm1)
    h       = DecoderGRU(d_D, return_sequences=True)([ye_tm1, c])
    s       = Maxout(d_C)([h, ye_tm1, RepeatVector(maxlen)(c)])
    s       = Dropout(.2)(s)
    c_I     = ProjectionOverTime(V_C.size)(s)

    super(W2C, self).__init__(input=[c, y_tm1], output=c_I, name='W2C')


"""
from keras.engine import Input
import numpy as np

m = W2C(4, 3, 4, 5, type("", (), dict(size=5))())
m.compile(optimizer='sgd', loss='mse')

X = np.array([[0, 1, 1, 2],
              [0, 1, 1, 1]], dtype=np.int32)


ye_tm1 = K.function([m.layers[2].input],
                    [m.layers[2].output])


res = ye_tm1([X])[0]
assert not np.any(res[0][0]), res[0][0]
assert not np.any(res[1][0]), res[1][0]
raise "Done"
"""
