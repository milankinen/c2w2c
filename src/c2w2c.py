import sys
import model_params

from time import strftime, localtime
from keras.layers import TimeDistributed, Input, Activation
from keras.models import Model
from keras.optimizers import Adam

from models import C2W, LanguageModel, W2C
from util import load_training_data, load_test_data, info, Timer
from validation import calc_perplexity


sys.setrecursionlimit(40000)

params = model_params.from_cli_args()
params.print_params()

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
W_ctx   = TimeDistributed(C2W(params, V_C, V_W))(input)
w_np1   = LanguageModel(params, V_C, V_W, state_seq=False)(W_ctx)
output  = W2C(params, V_C, V_W)(w_np1)

c2w2c   = Model(input=input, output=Activation('softmax')(output))

# Separate ub-models for testing / perplexity
lm      = Model(input=input, output=LanguageModel(params, V_C, V_W, state_seq=True)(W_ctx))
w2c_in  = Input(shape=(params.d_W,))
w2c     = Model(input=w2c_in, output=Activation('softmax')(W2C(params, V_C, V_W)(w2c_in)))


def update_weights():
  lm_model      = c2w2c.layers[2]
  w2c_model     = c2w2c.layers[3]
  lm.layers[2].set_weights(lm_model.get_weights())
  w2c.layers[1].set_weights(w2c_model.get_weights())


def delta_str(cur, prev):
  return '(%s%f)' % ('-' if cur < prev else '+', abs(prev - cur)) if prev is not None else ''


def test_model():
  total_pp = 0.0
  # loop all test sentences from the test set
  for expected, x in Xt:
    # get word embedding predictions to tested the sentence
    S_e = lm.predict(x)[0]
    # calculate perplexity for the sentence
    pp = calc_perplexity(V_W, V_Wt, V_C, expected, w2c.predict(S_e))
    total_pp += pp
  return total_pp / len(Xt)


print 'Compiling models...'
c2w2c.compile(optimizer=Adam(lr=0.001, epsilon=1e-10), loss='categorical_crossentropy', metrics=['accuracy'])
# optimizers are not important in LM and W2C because these models are used
# only for data validation, thus optimizer never gets used
lm.compile(optimizer='sgd', loss='mse')
w2c.compile(optimizer='sgd', loss='categorical_crossentropy')
print 'Compiled'


fit_t   = Timer()
test_t  = Timer()

prev_pp   = None
prev_loss = None
prev_acc  = None

try:
  print 'Training model...'
  for e in range(0, params.n_epoch):
    fit_t.start()
    epoch = e + 1
    print '=== Epoch %d ===' % epoch
    h = c2w2c.fit_generator(generator=training_data.as_generator(params.n_context, params.n_batch),
                            samples_per_epoch=training_data.get_num_samples(params.n_context),
                            nb_epoch=1,
                            verbose=1)
    fit_elapsed, fit_tot = fit_t.lap()
    update_weights()

    test_t.start()
    loss  = h.history['loss'][0]
    acc   = h.history['acc'][0]
    pp    = test_model()
    test_elapsed, test_tot = test_t.lap()

    epoch_info = '''Epoch %d summary at %s:
  - Model loss:         %f %s
  - Model accuracy:     %f %s
  - Model perplexity:   %f %s
  - Training took:      %s
  - Validation took:    %s
  - Total training:     %s
  - Total validation:   %s''' % (epoch, strftime("%Y-%m-%d %H:%M:%S", localtime()), loss, delta_str(loss, prev_loss),
                                 acc, delta_str(acc, prev_acc), pp, delta_str(pp, prev_pp), fit_elapsed, test_elapsed,
                                 fit_tot, test_tot)
    print ''
    info(epoch_info)

    prev_acc  = acc
    prev_loss = loss
    prev_pp   = pp

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

