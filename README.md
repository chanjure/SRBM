# RBM
---
** Scalar Field RBM **
---

## Usage
run:

  $ bash bin/run_train.sh

to run a sample training.
Resulting images are created into images/

for interactive example of using SRBM and setting hyperparameters, see notebook/SRBM_unsup_gen.

After modifying notebook example, use:

  $ bash bin/run_check.sh

before uploading to git

## TODO

## Directory structure
* Source code is in RBM/
* New features under dev is in RBM/test/
* Once the code under RBM/test/ is tested and implemented, the old code is transferred to RBM.old/
