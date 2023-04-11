# RBM
---
**Scalar Field RBM**
---

[![Run tests](https://github.com/chanjure/SRBM/actions/workflows/pytest.yaml/badge.svg?event=push)](https://github.com/chanjure/SRBM/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/chanjure/SRBM/branch/main/graph/badge.svg?token=2KRU5F0ERW)](https://codecov.io/gh/chanjure/SRBM)

## Usage
run:

  $ bash bin/run_train.sh

to run a sample training.
Resulting images are created into images/

for interactive example of using SRBM and setting hyperparameters, see notebook/SRBM_unsup_gen.

Before uploading to git, run:

  $ bash bin/run_check.sh

To run pytest and test notebook.

Or for automated push, run:

  $ bash bin/git_push.sh <comment>

For conda users, use:
  
  $ conda env create -f environment.yml -n qftml

or for pip users,
  
  $ python -m pip install -r requirements.txt

to setup the environment

## TODO

## Directory structure
* Source code is in RBM/
* New features under dev is in RBM/test/
* Once the code under RBM/test/ is tested and implemented, the old code is transferred to RBM.old/
