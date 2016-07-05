# C2W2C Language Model

The implementation of the language model from my Master's Thesis.


## Pre-requirements

In order to run the model, the following pre-requirements must be satisfied.

1. Install Python 2.7.x, `pip` and `virtualenv`

2. Create new virtual environment and install the following packages:

```bash
# Theano and keras and their deps
pip install numpy scipy pyyaml
pip install https://github.com/Theano/archive/rel-0.8.2.zip
pip install https://github.com/fchollet/keras/archive/1.0.5.zip

# If you want to load/save weights, install
pip install Cython
pip install h5py
```

**NOTE:** If you are using OSX, these native packages must be installed before you 
can install the actual Python packages (using `homebrew`)
```bash
# if gfortran is missing
brew install gcc 
# if you want to load/save weights
brew tap homebrew/science
brew install hdf5
```  

## Running the model

```bash
DEVICE=cpu ./run --training data/training.txt --test data/test.txt -w 21
```

Help about the available options:
```bash
DEVICE=cpu ./run -h
```

