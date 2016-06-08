
from models import C2W, LanguageModel, W2C
from util import load_training_data
from keras.models import Model
from keras.layers import TimeDistributed, Input, Activation


N_batch   = 50
N_ctx     = 10
d_C       = 150
d_W       = 50
d_Wi      = 150


training_data = load_training_data('data/training.txt')
V_C           = training_data.V_C
V_W           = training_data.V_W


# The actual C2W2C model
input   = Input(shape=(None, V_W.dim[1]), dtype='int32')
W_ctx   = TimeDistributed(C2W(V_C=V_C, V_W=V_W, d_C=d_C, d_W=d_W, d_Wi=d_Wi))(input)
w_np1   = LanguageModel(d_W, state_seq=False)(W_ctx)
output  = W2C(V_C=V_C, V_W=V_W, d_W=d_W, d_C=d_C)(w_np1)

c2w2c   = Model(input=input, output=Activation('softmax')(output))

print 'Compiling model...'
c2w2c.compile(optimizer='adam', loss='categorical_crossentropy')
print 'Compiled'

try:
  print 'Training model...'
  c2w2c.fit_generator(generator=training_data.as_generator(N_ctx, N_batch),
                      samples_per_epoch=training_data.get_num_samples(N_ctx),
                      nb_epoch=1,
                      verbose=1)
  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

