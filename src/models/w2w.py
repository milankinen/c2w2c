from keras.layers import Input
from keras.models import Model
from ..layers import Projection


class W2W(Model):
  def __init__(self, n_batch, d_L, V_W, trainable=True):
    """
      n_batch  :: batch size for model application
      maxlen   :: maximum sampled word length
      d_L      :: language model state dimension (input embedding vector size)
      d_D      :: internal decoder state dimension
      V_C      :: character vocabulary
    """

    w_np1E  = Input(batch_shape=(n_batch, d_L), name='w_np1e', dtype='floatX')
    w_np1W  = Projection(V_W.size, trainable=trainable)(w_np1E)

    super(W2W, self).__init__(input=w_np1E, output=w_np1W, name='W2W')
