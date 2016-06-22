from keras.layers import LSTM, Input, Dense, Masking, merge
from keras.models import Model
from ..layers import ProjectionOverTime


class C2W(Model):
  def __init__(self, maxlen, d_C, d_W, d_Wi, V_C, trainable=True):
    """
      maxlen   :: maximum input word length
      d_C      :: character features (input embedding vector size)
      d_W      :: word features (output word embedding vector size)
      d_Wi     :: internal encoder state dimension
      V_C      :: character vocabulary
    """

    c_I       = Input(shape=(maxlen, V_C.size), dtype='int8')
    c_E       = ProjectionOverTime(d_C, trainable=trainable)(Masking(0.)(c_I))

    forward   = LSTM(d_Wi,
                     return_sequences=False,
                     go_backwards=False,
                     consume_less='gpu')(c_E)
    backwards = LSTM(d_Wi,
                     return_sequences=False,
                     go_backwards=True,
                     consume_less='gpu')(c_E)

    s_Ef      = Dense(d_W, trainable=trainable)(forward)
    s_Eb      = Dense(d_W, trainable=trainable)(backwards)
    s_E       = merge(inputs=[s_Ef, s_Eb], mode='sum')

    super(C2W, self).__init__(input=c_I, output=s_E, name='C2W')

