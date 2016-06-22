import os.path as path
import sys
from time import strftime, localtime

import numpy as np
from keras.models import Model
from keras.optimizers import Adam

import model_params
from c2w2c import C2W2C
from c2w2w import C2W2W
from dataset import load_dataset, make_char_vocabulary
from training import prepare_c2w2c_training_data, prepare_c2w2w_training_data
from util import info, Timer, MiniIteration
from validation import make_c2w2c_test_function, make_c2w2w_test_function

sys.setrecursionlimit(40000)


params = model_params.from_cli_args()
params.print_params()

print 'Loading training data...'
training_dataset = load_dataset(params.training_dataset, params.train_data_limit)
training_dataset.print_stats()

print 'Loading test data...'
test_dataset = load_dataset(params.test_dataset, params.test_data_limit)
test_dataset.print_stats()

# Vocabularies
V_C = make_char_vocabulary([test_dataset, training_dataset])
V_W = training_dataset.vocabulary

print 'V_C statistics:'
print '  - Distinct characters: %d' % V_C.size


def compile_model(model, returns_chars=True):
  lr   = params.learning_rate
  clip = 5.
  adam = Adam(lr=lr, clipvalue=clip)
  model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                sample_weight_mode=('temporal' if returns_chars else None),
                metrics=['accuracy'])


def build_c2w2c_validation_models():
  _, (c2w, lm, w2c), inputs = C2W2C(1, params, V_C)
  w_nc, w_nmask, w_np1c     = inputs

  w_nE      = c2w(w_nc)
  w_np1E    = lm([w_nE, w_nmask])
  c2wp1     = Model(input=[w_nc, w_nmask], output=w_np1E)

  compile_model(c2wp1, returns_chars=False)
  compile_model(w2c, returns_chars=True)

  return c2wp1, w2c


def load_weights(model, filename):
  if filename:
    if path.isfile(filename):
      print 'Loading existing weights from "%s"...' % filename
      model.load_weights(filename)
    else:
      print 'Initial weight file not found: %s' % filename


def param_count(m):
  return sum([w.size for w in m.get_weights()])


def prepare_env(mode):
  if mode == 'c2w2c':
    # Train c2w2c model as it is
    trainable_model, (c2w, lm, w2c), _  = C2W2C(params.n_batch, params, V_C)
    v_c2wp1, v_w2c                      = build_c2w2c_validation_models()
    compile_model(trainable_model, returns_chars=True)
    load_weights(trainable_model, params.init_weight_file)

    #print trainable_model.layers
    def update_weights():
      v_c2wp1.layers[-2].set_weights(c2w.get_weights())
      v_c2wp1.layers[-1].set_weights(lm.get_weights())
      v_w2c.set_weights(w2c.get_weights())

    test_model        = make_c2w2c_test_function(v_c2wp1, v_w2c, params, test_dataset, V_C, V_W)
    training_data     = prepare_c2w2c_training_data(params, training_dataset, V_C)

    print 'Model parameters:'
    print ' - C2W:%10s' % str(param_count(c2w))
    print ' - LM: %10s' % str(param_count(lm))
    print ' - W2C:%10s' % str(param_count(w2c))
    print '       %s' % ('-' * 10)
    print '       %10s' % str(sum([param_count(m) for m in [c2w, lm, w2c]]))

  elif mode == 'w2c_train':
    """
    # Load existing weights from previously trained word model, set C2W
    # and LM weights and train only W2C weights
    trainable_model   = build_trainable_model(params.n_batch, mode='c2w2c', c2w_trainable=False, lm_trainable=False, w2c_trainable=True)
    c2w2w_model       = build_trainable_model(params.n_batch, mode='word')
    lm, w2c           = build_validation_models('c2wc2')
    lm_idx            = -3
    update_weights    = lambda: (lm.set_weights([w for l in trainable_model.layers[0:-2] for w in l.get_weights()]),
                                 w2c.set_weights(trainable_model.layers[-1].get_weights()))
    test_model        = make_c2w2c_test_function(lm, lambda: reset_lm(lm.layers[-1]), w2c, params, test_dataset, V_C, V_W)
    training_data     = prepare_c2w2c_training_data(params, training_dataset, V_C)
    load_weights(c2w2w_model, params.init_weight_file)
    print 'Transferring C2W2W model weights to C2W2C...'
    for i in range(0, len(c2w2w_model.layers) - 1):
      dest = trainable_model.layers[i]
      src  = c2w2w_model.layers[i]
      dest.set_weights(src.get_weights())
    """
    raise NotImplementedError
  elif mode == 'c2w2w':
    trainable_model   = C2W2W(params.n_batch, params, V_C, V_W)
    validation_model  = C2W2W(1, params, V_C, V_W)
    compile_model(trainable_model, returns_chars=False)
    compile_model(validation_model, returns_chars=False)
    load_weights(trainable_model, params.init_weight_file)

    def update_weights():
      validation_model.set_weights(trainable_model.get_weights())

    test_model        = make_c2w2w_test_function(validation_model, params, test_dataset, V_C, V_W)
    training_data     = prepare_c2w2w_training_data(params, training_dataset, V_C, V_W)

    print 'Model parameters:'
    print ' - Total:%10s' % str(param_count(trainable_model))

  else:
    print 'Invalid mode: %s' % mode
    sys.exit(1)

  def test_fn(limit=None):
    update_weights()
    return test_model(limit)

  return trainable_model, test_fn, training_data


MODE = params.mode
model, run_tests, t_data = prepare_env(MODE)


def delta_str(cur, prev, fmt='(%s%f)'):
  return fmt % ('-' if cur < prev else '+', abs(prev - cur)) if prev is not None else ''

fit_t   = Timer()
test_t  = Timer()

prev_pp   = None
prev_loss = None
prev_acc  = None


def run_model_tests(prev_pp):
  if params.gen_n_samples is not None:
    print 'Text generation currently not supported...'    # TODO: implement

  print 'Validating model...'
  test_t.start()
  pp, oov = run_tests()
  test_elapsed, test_tot = test_t.lap()
  validation_info = '''Validation results:
  - Perplexity:         %3g %s
  - OOV rate:           %f
  - Validation took:    %s
  - Total validation:   %s''' % (pp, delta_str(pp, prev_pp, '(%s%3g)'), oov, test_elapsed, test_tot)
  print ''
  info(validation_info)
  return pp, oov


best_weights = None

try:
  if params.test_only:
    run_model_tests(None)
    sys.exit(0)

  print 'Training model...'
  for e in range(0, params.n_epoch):
    fit_t.start()
    epoch = e + 1
    print '=== Epoch %d ===' % epoch

    n_samples, make_gen = t_data
    sentence_seq, data_generator = make_gen()

    mini_iter = MiniIteration(prev_pp, sentence_seq, model, run_tests,
                              run_minitest_after=params.mini_iteration)
    model.reset_states()
    h = model.fit_generator(generator=data_generator,
                            samples_per_epoch=n_samples,
                            callbacks=[mini_iter],
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
    pp, _ = run_model_tests(prev_pp)

    if prev_pp is None or pp <= prev_pp:
      best_weights = np.array(model.get_weights())
      prev_acc  = acc
      prev_loss = loss
      prev_pp   = pp
      if params.save_weight_file:
        filename = params.save_weight_file  #'%s.%d' % (params.save_weight_file, (e % 10) + 1)
        model.save_weights(filename, overwrite=True)
        info('Model weights saved to %s' % filename)
    else:
      print 'Perplexity didn\'t improve. Resetting weights...'
      model.set_weights(best_weights)

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

