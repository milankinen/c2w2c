import os.path as path
import sys
from time import strftime, localtime

import numpy as np
from keras.optimizers import Adam

import model_params
from c2w2c import C2W2C, build_c2w2c_validation_models
from c2w2w import C2W2W
from models import W2C
from datagen import prepare_c2w2c_training_data, prepare_c2w2w_training_data, prepare_w2c_training_data
from dataset import load_dataset, make_char_vocabulary
from util import info, Timer
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


def try_load_weights(model, filename):
  if filename:
    if path.isfile(filename):
      print 'Loading existing weights from "%s"...' % filename
      model.load_weights(filename)
    else:
      print 'Initial weight file not found: %s' % filename


def try_save_weights(model, filename):
  if filename:
    model.save_weights(filename, overwrite=True)
    print 'Weights saved to: %s' % filename


def param_count(m):
  return sum([w.size for w in m.get_weights()])


def delta_str(cur, prev, fmt='(%s%f)'):
  return fmt % ('-' if cur < prev else '+', abs(prev - cur)) if prev is not None else ''


def c2w2c_from_c2w2w_weights(c2w2w_weights_file):
  def weight_list(m):
    return list([w.tolist() for w in m.get_weights()])

  c2w2c, (c2w, lm, w2c) = C2W2C(params.n_batch, params, V_C)
  c2w2w                 = C2W2W(params.n_batch, params, V_C, V_W)
  compile_model(c2w2c, returns_chars=True)
  compile_model(c2w2w, returns_chars=False)
  try_load_weights(params.init_weight_file)
  try_load_weights(c2w2w_weights_file)

  prev_w    = weight_list(c2w2c)
  prev_w2cw = weight_list(w2c)
  assert prev_w == weight_list(c2w2c)
  for i in range(0, len(c2w2w.layers) - 2):
    print c2w2c.layers[i]
    c2w2c.layers[i].set_weights(c2w2w.layers[i].get_weights())
  assert prev_w != weight_list(c2w2c)
  assert prev_w2cw != weight_list(w2c)
  return c2w2c, (c2w, lm, w2c)


def prepare_env(mode):
  if mode == 'c2w2c':
    # Train c2w2c model as it is
    trainable_model, (c2w, lm, w2c), _  = C2W2C(params.n_batch, params, V_C)
    v_c2wp1, v_w2c                      = build_c2w2c_validation_models(params, V_C)
    compile_model(trainable_model, returns_chars=True)
    compile_model(v_c2wp1, returns_chars=False)
    compile_model(v_w2c, returns_chars=True)
    try_load_weights(trainable_model, params.init_weight_file)

    def update_weights():
      v_c2wp1.set_weights(c2w.get_weights() + lm.get_weights())
      v_w2c.set_weights(w2c.get_weights())

    def save_weights():
      try_save_weights(trainable_model, params.save_weight_file)

    test_model        = make_c2w2c_test_function(trainable_model, v_c2wp1, v_w2c, params, test_dataset, V_C, V_W)
    training_data     = prepare_c2w2c_training_data(params, training_dataset, V_C)

    print 'Model parameters:'
    print ' - C2W:%10s' % str(param_count(c2w))
    print ' - LM: %10s' % str(param_count(lm))
    print ' - W2C:%10s' % str(param_count(w2c))
    print '       %s' % ('-' * 10)
    print '       %10s' % str(sum([param_count(m) for m in [c2w, lm, w2c]]))

  elif mode == 'c2w2w':
    trainable_model   = C2W2W(params.n_batch, params, V_C, V_W)
    compile_model(trainable_model, returns_chars=False)
    compile_model(validation_model, returns_chars=False)
    try_load_weights(trainable_model, params.init_weight_file)

    def update_weights():
      pass  # using same model for test as used in training

    def save_weights():
      try_save_weights(trainable_model, params.save_weight_file)

    test_model        = make_c2w2w_test_function(validation_model, params, test_dataset, V_C, V_W)
    training_data     = prepare_c2w2w_training_data(params, training_dataset, V_C, V_W)

    print 'Model parameters:'
    print ' - Total:%10s' % str(param_count(trainable_model))

  elif mode == 'w2c':
    if not params.c2w2w_weights:
      print 'C2W2W weights are mandatory when training W2C model'
      sys.exit(1)

    # build models: w2c for training, validation models for data generation and quick PP checks
    trainable_model       = W2C(params.n_batch, params.maxlen, params.d_W, params.d_D, V_C, apply_softmax=True)
    c2w2c, (c2w, lm, w2c) = c2w2c_from_c2w2w_weights(params.c2w2w_weights)
    c2wp1, _              = build_c2w2c_validation_models(params, V_C)
    compile_model(trainable_model, returns_chars=True)
    compile_model(c2wp1, returns_chars=False)
    c2wp1.set_weights(c2w.get_weights() + lm.get_weights())
    trainable_model.set_weights(w2c.get_weights())

    def update_weights():
      # c2w2c needs updated weights if validating with "full" mode
      w2c.set_weights(trainable_model.get_weights())

    def save_weights():
      update_weights()
      try_save_weights(c2w2c, params.save_weight_file)

    training_data = prepare_w2c_training_data(c2wp1, params, training_dataset, V_C)
    test_model    = make_c2w2c_test_function(c2w2c, c2wp1, trainable_model, params, test_dataset, V_C, V_W)

  else:
    print 'Invalid mode: %s' % mode
    sys.exit(1)

  def test_fn(limit=None):
    update_weights()
    return test_model(limit)

  return trainable_model, test_fn, save_weights, training_data


fit_t     = Timer()
test_t    = Timer()

prev_pp   = None
prev_loss = None
prev_acc  = None

model, run_tests, persist_weights, t_data = prepare_env(params.mode)


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

    n_samples, data_generator = t_data

    model.reset_states()
    h = model.fit_generator(generator=data_generator,
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
    pp, _ = run_model_tests(prev_pp)

    if prev_pp is None or pp <= prev_pp:
      best_weights = np.array(model.get_weights())
      prev_acc  = acc
      prev_loss = loss
      prev_pp   = pp
      persist_weights()

  print 'Training complete'
except KeyboardInterrupt:
  print 'Training interrupted. Bye'

