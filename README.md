## HAM10000 Skin Lesion Classifier
This classifier was developed for my Bachelor's Thesis. 

### DATA SETUP
Download the [train](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and [test](https://kaggle.com/datasets/991ec8f5e527dd9009c8e524b5572622f02cf0d605365cab2fa424d2256bddcf) sets.
These are expected in `data_dir`, `./data` by default.
 
### Training
Training command
```
python train.py --data-dir [./data] --train-folder [skin-cancer-mnist-ham10000] --model-dir [./models] --model-type --init-lr [0.001] --batch-size [64] --n-epochs [15] --scheduler-type [NONE] --weight-method [NONE] --sampling-method [NONE] --snapshots
```
Models are saved in `./models` by default.
Model snapshots are saved in this convention: 
`[model_type]_TRAIN_[init_lr]_[batch_size]_[n_epochs]_[scheduler_type]_[weight_method]_[sampling_method]_[snapshot_epoch]`
or, if not a snapshot:
`[model_type]_TRAIN_[init_lr]_[batch_size]_[n_epochs]_[scheduler_type]_[weight_method]_[sampling_method]`



### Validating
To test different techniques with k-fold cross validation:
```
python validate.py --data-dir [./data] --train-folder [skin-cancer-mnist-ham10000] --model-dir [./models] --model-type --init-lr [0.001] --batch-size [64] --n-epochs [15] --scheduler-type [NONE] --weight-method [NONE] --sampling-method [NONE] --n-splits [5]
```

Validation results are printed out. Early stopping is used for BMCA and accuracy per fold. Results and models ARE NOT SAVED.

### Testing
To test on the test set, models to be tested in the ensemble must be saved to `ensemble_dir` which is `./ensemble` by default.
Test results are printed out. 

```
python test.py --data-dir [./data] --test-folder [ham10000-test] --ensemble-dir [./ensemble] --batch-size [64] --input-size [224]
```

### A lightweight ensemble of EfficientNets

The current models in `./ensemble` are the final 82.1% BMCA nad 84.2% accuracy with a ~30ms inference time using a P100 GPU
