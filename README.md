# C2W2C Language Model

The implementation of the language model from my Master's Thesis. If you are interested
in getting the paper, please send me email to m.lankinen@iki.fi.


## Pre-requirements RAVAL VIVEK

In order to run the model, the following pre-requirements must be satisfied.

1. Install Python 2.7.x, `pip` and `virtualenv`

2. Create new virtual environment and install the following packages:

```bash
# Theano and keras and their deps
pip install numpy scipy pyyaml
pip install https://github.com/Theano/archive/rel-0.8.2.zip
pip install https://github.com/fchollet/keras/archive/1.0.5.zip
pip install Cython
pip install h5py
```

**NOTE:** If you are using OSX, these native packages must be installed before you 
can install the actual Python packages (using `homebrew`)
```bash
# if gfortran is missing
brew install gcc 
brew tap homebrew/science
brew install hdf5
```  

## Running the model

Running C2W2C model and use example data from `data` folder:
```bash
./run_c2w2c.sh
```

Running WordLSTM model and use example data from `data` folder:
```bash
./run_word_lstm.sh
```

**NOTE**: If you are using OS X and XCode 7.x, you may need to install older version
of XCode and set `$DEVELOPER_DIR` environment variable to point to older installation
(see example from `run` script).


### Available options

```
Usage: ./run <args>

C2W2C language model

optional arguments:
  -h, --help            show this help message and exit
  --training filename   Training dataset filename
  --test filename       Validation dataset filename
  --data-limit training:validation, -l training:validation
                        Limit data size to the given rows (e.g. "10:1")
  --batch-size n        Number of samples is single training batch
  --learning-rate num, -r num
  --num-epoch n, -e n   Number of epoch to run
  --load-weights filename
                        File containing the initial model weights
  --save-weights filename
                        Filename where model weights will be saved
  --max-word-length n, -w n
                        Maximum word length (longer words will be truncated)
  --d_C n               Character features vector size
  --d_W n               Word features vector size
  --d_Wi n              Intermediate word LSTM state dimension
  --d_L n               Language model state dimension
  --d_D n               W2C Decoder state dimension
  --gen-text n          Generate N sample sentences after each epoch
  --test-only, -T       Run only PP test and (optional) text generation
  --mode c2w2c|word     Select which mode to run
```

## Example data

Example data files are taken from [Europarl V7](http://www.statmt.org/europarl/) Finnish 
Corpus and pre-processed with [Apache OpenNLP](https://opennlp.apache.org/) tokenizer and
[Finnish tokenizing model](https://github.com/TurkuNLP/Finnish-dep-parser).

* `training.txt` : 700 sentences / 15k tokens
* `validation.txt` : 30 sentences / 600 tokens

## License 

MIT
