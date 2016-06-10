import sys, os.path as path
import model_params

from time import strftime, localtime
from keras.layers import TimeDistributed, Input, Activation
from keras.models import Model
from keras.optimizers import Adam

from dataset import make_training_samples_generator, make_test_samples, load_dataset, make_char_vocabulary
from models import C2W, LanguageModel, W2C
from util import info, Timer
from validation import test_model


sys.setrecursionlimit(40000)


params = model_params.from_cli_args()
params.print_params()

print 'Loading training data...'
training_data = load_dataset(params.training_dataset, 50)
training_data.print_stats()

print 'Loading test data...'
test_data = load_dataset(params.test_dataset, 1)
test_data.print_stats()

# Vocabularies
V_C = make_char_vocabulary([test_data, training_data])

# Test samples
test_samples = make_test_samples(params, test_data, V_C)


# The actual C2W2C model
input   = Input(shape=(None, params.maxlen), dtype='int32')
W_ctx   = TimeDistributed(C2W(params, V_C))(input)
w_np1   = LanguageModel(params, V_C, state_seq=False)(W_ctx)
output  = W2C(params, V_C)(w_np1)

c2w2c   = Model(input=input, output=Activation('softmax')(output))

# Separate ub-models for testing / perplexity
lm      = Model(input=input, output=LanguageModel(params, V_C, state_seq=True)(W_ctx))
w2c_in  = Input(shape=(params.d_W,))
w2c     = Model(input=w2c_in, output=Activation('softmax')(W2C(params, V_C)(w2c_in)))


def update_weights():
  lm_model      = c2w2c.layers[2]
  w2c_model     = c2w2c.layers[3]
  lm.layers[2].set_weights(lm_model.get_weights())
  w2c.layers[1].set_weights(w2c_model.get_weights())


def delta_str(cur, prev):
  return '(%s%f)' % ('-' if cur < prev else '+', abs(prev - cur)) if prev is not None else ''


print 'Compiling models...'
c2w2c.compile(optimizer=Adam(lr=params.learning_rate, epsilon=1e-10), loss='categorical_crossentropy', metrics=['accuracy'])
# optimizers are not important in LM and W2C because these models are used
# only for data validation, thus optimizer never gets used
lm.compile(optimizer='sgd', loss='mse')
w2c.compile(optimizer='sgd', loss='categorical_crossentropy')
print 'Compiled'

if params.init_weight_file:
  if path.isfile(params.init_weight_file):
    print 'Loading existing weights from "%s" ...' % params.init_weight_file
    c2w2c.load_weights(params.init_weight_file)
  else:
    print 'Initial weight file not found: %s' % params.init_weight_file

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
    gen, n_samples = make_training_samples_generator(params, training_data, V_C)
    h = c2w2c.fit_generator(generator=gen,
                            samples_per_epoch=n_samples,
                            nb_epoch=1,
                            verbose=1)
    fit_elapsed, fit_tot = fit_t.lap()

    # needed for validation models LM and W2C
    update_weights()

    test_t.start()
    loss    = h.history['loss'][0]
    acc     = h.history['acc'][0]
    pp, oov = test_model(params, lm, w2c, test_samples, test_data.vocabulary, V_C)
    test_elapsed, test_tot = test_t.lap()

    epoch_info = '''Epoch %d summary at %s:
  - Model loss:         %f %s
  - Model accuracy:     %f %s
  - Model perplexity:   %f %s
  - OOV rate:           %f
  - Training took:      %s
  - Validation took:    %s
  - Total training:     %s
  - Total validation:   %s''' % (epoch, strftime("%Y-%m-%d %H:%M:%S", localtime()), loss, delta_str(loss, prev_loss),
                                 acc, delta_str(acc, prev_acc), pp, delta_str(pp, prev_pp), oov, fit_elapsed,
                                 test_elapsed, fit_tot, test_tot)
    print ''
    info(epoch_info)

    if params.save_weight_file and (prev_loss is None or prev_loss > loss):
      c2w2c.save_weights(params.save_weight_file, overwrite=True)
      print 'Model weights saved'

    prev_acc  = acc
    prev_loss = loss
    prev_pp   = pp

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

