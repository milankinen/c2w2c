import sys, os.path as path, numpy as np
import model_params

from time import strftime, localtime
from keras.layers import TimeDistributed, Input
from keras.models import Model
from keras.optimizers import Adam

from dataset import make_training_samples_generator, make_test_samples, load_dataset, make_char_vocabulary, make_dataset_from_sentence
from layers import FastDistribute
from models import C2W, LanguageModel, W2C
from util import info, Timer
from validation import test_model, gen_text


sys.setrecursionlimit(40000)


params = model_params.from_cli_args()
params.print_params()

print 'Loading training data...'
training_data = load_dataset(params.training_dataset, params.train_data_limit)
training_data.print_stats()

print 'Loading test data...'
test_data = load_dataset(params.test_dataset, params.test_data_limit)
test_data.print_stats()

# Vocabularies
V_C = make_char_vocabulary([test_data, training_data])

print 'V_C statistics:'
print '  - Distinct characters: %d' % V_C.size

# Test samples
print 'Preparing test samples...'
test_samples = make_test_samples(params, test_data, V_C)


# The actual C2W2C model
print 'Defining models...'
ctx_in    = Input(shape=(params.n_context, params.maxlen, V_C.size), dtype='int8', name='context')
pred_in   = Input(shape=(params.maxlen, V_C.size), dtype='int8', name='predicted_word')
W_ctx     = FastDistribute(C2W(params, V_C))(ctx_in)
w_np1     = LanguageModel(params, V_C, state_seq=False)(W_ctx)
C_I       = W2C(params, V_C, p_input=pred_in)([w_np1, pred_in])

c2w2c     = Model(input=[ctx_in, pred_in], output=C_I)

# Separate sub-models for testing / perplexity
lm_Cin    = Input(shape=(None, params.maxlen, V_C.size), dtype='int8', name='context')
lm_out    = LanguageModel(params, V_C, state_seq=True)(TimeDistributed(C2W(params, V_C))(lm_Cin))
lm        = Model(input=lm_Cin, output=lm_out)
w2c_Ein   = Input(shape=(params.d_W,), dtype='floatX', name='embedding')
w2c_Pin   = Input(shape=(params.maxlen, V_C.size), dtype='int8', name='predicted_word')
w2c       = W2C(params, V_C, e_input=w2c_Ein, p_input=w2c_Pin)


def update_weights():
  lm_weights    = [w for i in range(0, 3) for w in c2w2c.layers[i].get_weights()]
  w2c_weights   = c2w2c.layers[4].get_weights()
  lm.set_weights(lm_weights)
  w2c.set_weights(w2c_weights)


def generate_sample_sentences(n_samples):
  sample_seeds = []
  V_W          = training_data.vocabulary
  while len(sample_seeds) < n_samples:
    i = np.random.randint(0, len(training_data.sentences))
    sent = training_data.sentences[i]
    if len(sent) > 5:
      sample_seeds.append(sent[0: 5])
      continue
  gen_text(params, lm, w2c, sample_seeds, V_W, V_C, n_words=15)


def delta_str(cur, prev):
  return '(%s%f)' % ('-' if cur < prev else '+', abs(prev - cur)) if prev is not None else ''


print 'Compiling models...'
c2w2c.compile(optimizer=Adam(lr=params.learning_rate),
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=['accuracy'])
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


n_params_c2w  = sum([np.prod(np.array(w.shape)) for w in c2w2c.layers[1].get_weights()])
n_params_lm   = sum([np.prod(np.array(w.shape)) for w in c2w2c.layers[2].get_weights()])
n_params_w2c  = sum([np.prod(np.array(w.shape)) for w in c2w2c.layers[4].get_weights()])

print 'Model parameters:'
print '  - C2W:             %d' % n_params_c2w
print '  - Language Model:  %d' % n_params_lm
print '  - W2C:             %d' % n_params_w2c
print '-----------------------'
print '             Total:  %d' % sum([n_params_c2w, n_params_lm, n_params_w2c])
print ''


fit_t   = Timer()
test_t  = Timer()

prev_pp   = None
prev_loss = None
prev_acc  = None


def run_model_tests():
  if params.gen_n_samples is not None:
    print 'Generating %d sample sentences...' % params.gen_n_samples
    generate_sample_sentences(params.gen_n_samples)
    print ''

  print 'Validating model...'
  test_t.start()
  pp, oov = test_model(params, lm, w2c, test_samples, test_data.vocabulary, V_C)
  test_elapsed, test_tot = test_t.lap()
  validation_info = '''Validation results:
  - Perplexity:         %f %s
  - OOV rate:           %f
  - Validation took:    %s
  - Total validation:   %s''' % (pp, delta_str(pp, prev_pp), oov, test_elapsed, test_tot)
  print ''
  info(validation_info)
  return pp, oov


try:
  if params.test_only:
    update_weights()
    run_model_tests()
    sys.exit(0)

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

    loss       = h.history['loss'][0]
    acc        = h.history['acc'][0]
    epoch_info = '''Epoch %d summary at %s:
  - Model loss:         %f %s
  - Model accuracy:     %f %s
  - Training took:      %s
  - Total training:     %s''' % (epoch, strftime("%Y-%m-%d %H:%M:%S", localtime()), loss, delta_str(loss, prev_loss),
                                 acc, delta_str(acc, prev_acc), fit_elapsed, fit_tot)
    print ''
    info(epoch_info)

    # needed for validation models LM and W2C
    update_weights()
    pp, _ = run_model_tests()

    if params.save_weight_file and (prev_loss is None or prev_loss > loss):
      filename = '%s.%d' % (params.save_weight_file, (e % 10) + 1)
      c2w2c.save_weights(filename, overwrite=True)
      info('Model weights saved to %s' % filename)

    prev_acc  = acc
    prev_loss = loss
    prev_pp   = pp

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

