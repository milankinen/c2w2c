from keras import backend as K
from keras.engine.topology import Layer


class ContextMask(Layer):
  def __init__(self, **kwargs):
    super(ContextMask, self).__init__(**kwargs)

  def call(self, x, mask=None):
    assert mask is None
    return K.cast(K.any(x, axis=-1), K.floatx())

  def get_output_shape_for(self, input_shape):
    return (input_shape[0],)


"""
from keras.engine import Model, Input
import numpy as np

inp = Input(batch_shape=(2, 3, 4))

m = Model(inp, ContextMask()(inp))
m.compile(optimizer='sgd', loss='mse')

X = np.array([[[1, 1, 2, 3],
               [2, 2, 3, 0],
               [0, 0, 0, 0]]] * 2)

res = m.predict(X, batch_size=2).tolist()
assert res == [[1., 1., 0]] * 2
raise "Done"
"""
