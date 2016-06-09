import sys

from models import C2W, LanguageModel, W2C
from util import load_training_data, load_test_data, calc_perplexity
from keras.models import Model
from keras.layers import TimeDistributed, Input, Activation
from keras.optimizers import Adam


sys.setrecursionlimit(40000)


N_batch   = 50
N_ctx     = 5
N_epoch   = 300


d_C       = 50
d_W       = 50
d_Wi      = 150


print 'Loading test data...'
test_data = load_test_data('data/test.txt')
test_data.print_stats()

print 'Loading training data...'
training_data = load_training_data('data/training.txt', test_data)
training_data.print_stats()

# Vocabularies
V_C   = training_data.V_C
V_W   = training_data.V_W
V_Wt  = test_data.V_W

# Test samples
Xt = training_data.make_test_sentences(test_data)


# The actual C2W2C model
input   = Input(shape=(None, V_W.maxlen), dtype='int32')
W_ctx   = TimeDistributed(C2W(V_C=V_C, V_W=V_W, d_C=d_C, d_W=d_W, d_Wi=d_Wi))(input)
w_np1   = LanguageModel(d_W, state_seq=False)(W_ctx)
output  = W2C(V_C=V_C, V_W=V_W, d_W=d_W, d_C=d_C)(w_np1)

c2w2c   = Model(input=input, output=Activation('softmax')(output))

# Separate ub-models for testing / perplexity
lm      = Model(input=input, output=LanguageModel(d_W, state_seq=True)(W_ctx))
w2c_in  = Input(shape=(d_W,))
w2c     = Model(input=w2c_in, output=Activation('softmax')(W2C(V_C=V_C, V_W=V_W, d_W=d_W, d_C=d_C)(w2c_in)))


def update_weights():
  lm_model      = c2w2c.layers[2]
  w2c_model     = c2w2c.layers[3]
  lm.layers[2].set_weights(lm_model.get_weights())
  w2c.layers[1].set_weights(w2c_model.get_weights())


def test_model():
  total_pp = 0.0
  # loop all test sentences from the test set
  for expected, x in Xt:
    # get word embedding predictions to tested the sentence
    S_e = lm.predict(x)[0]
    # calculate perplexity for the sentence
    pp = calc_perplexity(V_W, V_Wt, V_C, expected, w2c.predict(S_e))
    print pp
    total_pp += pp

  print 'PP = %f' % (total_pp / len(Xt))


print 'Compiling models...'
c2w2c.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy')
# optimizers are not important in LM and W2C because these models are used
# only for data validation, thus optimizer never gets used
lm.compile(optimizer='sgd', loss='mse')
w2c.compile(optimizer='sgd', loss='categorical_crossentropy')
print 'Compiled'

try:
  print 'Training model...'
  for e in range(0, N_epoch):
    print '=== Epoch %d ===' % e
    c2w2c.fit_generator(generator=training_data.as_generator(N_ctx, N_batch),
                        samples_per_epoch=training_data.get_num_samples(N_ctx),
                        nb_epoch=1,
                        verbose=1)

    update_weights()
    test_model()

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

