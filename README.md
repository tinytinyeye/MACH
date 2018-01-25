# MACH
MACH is a hash-based extreme multi-class classification package.  This package supports both sparse datasets and dense datasets. The training process is implemented in Tensorflow and supports GPU acceleration. Inference process consists of two stages: prediction stage and merging stage. In prediction stage, MACH uses Tensorflow to perform prediction for each meta classifier. In merging stage, MACH uses Numpy to merge results from all meta-classifiers. The merging utilizes python's multi processing module to achieve multi-core parallelization. GPU acceleration for merging stage will be supported in the future.

## Required installation
* Python 3
* Tensorflow
* Numpy

## Quickstart
Currently, MACH provides two demos: ODP and Imagenet. The following steps will show users how to download datasets and successfully run MACH on them.
### ODP
1. Download all files from [link](https://github.com/JohnLangford/vowpal_wabbit/tree/master/demo/recall_tree/odp)
2. Download datasets by typing `make odp_train.vw.gz` and `make odp_test.vw.gz:` in shell. Then unarchive `.gz` files to obtain `.vw` files.
4. Open `odp` folder and use the following script to convert datasets from `vw` format to `tfrecords` format: `python3 save_tfrecords.py vwFileName outputFileName`. To save the original training set as `training.tfrecords`, simply typing `python3 save_tfrecords.py odp_train.vw training.tfrecords`
5. After converting files to `tfrecords` format, change `TRAIN_FILE` and `TEST_FILE` fields in `odp_demo.py` to the location of your ODP datasets.
6. To start training and predicting ODP dataset, simply typing `python3 odp_demo.py -b 32 -r 50`. This line will start training for 50 meta-classifiers with 32 buckets. You may change the parameters to run different experiments.
### Imagenet
1. Download all files from [link](https://github.com/JohnLangford/vowpal_wabbit/tree/master/demo/recall_tree/imagenet)
2. Download datasets by typing `training.txt.gz` and `make testing.txt.gz` in shell. Then unarchive `.gz` files to obtain `.txt` files.
3. Open `imagenet` folder and use the following script to convert datasets from `txt` format to `tfrecords` format: `python3 save_tfrecords.py txtFileName outputFileName`.  To save the original training set as `training.tfrecords`, simply typing `python3 save_tfrecords.py training.txt training.tfrecords`. Both the source file and target file will be extremely large. Be sure to have enough disk space.
4. After converting files to `tfrecords` format, change `TRAIN_FILE` and `TEST_FILE` fields in `imagenet_demo.py` to the location of your imagenet datasets.
6. To start training and predicting ODP dataset, simply typing `python3 imagenet_demo.py -b 512 -r 20`. This line will start training for 20 meta-classifiers with 512 buckets. You may change the parameters to run different experiments.

##  Running MACH on other datasets
* By modifying source codes in `odp` or `imagenet` folders, users can run MACH on other large scale datasets.
### Sparse datasets
* The ODP dataset used in demo is a sparse dataset and therefore all the codes in `odp` folder is designed for sparse datasets.
* Because both training process and predicting process rely on Tensorflow and `tfrecords` format, before running MACH, users need to first convert their datasets to `tfrecords` format specified in `save_to_tfrecords` function in `odp/util.py`. This function essentially reads sparse format data line by line, stores indices and values separately for each data entry, and writes results into `tfrecords` format. **Feature index and label must starts from 0.**
* After the conversion finished, users will need to modify `NUM_FEATURES`, `NUM_CLASSES`, `TRAIN_FILE`, `TEST_FILE` in `odp_demo.py` to accommodate their datasets. If the user wishes to only perform training or predicting, the user can modify train_odp.py and predict_odp.py in a similar manner.
* Running MACH will be similar to the tutorials shown in Quickstart section.
### Dense datasets
* The Imagenet dataset used in demo is a dense dataset and therefore all the codes in `imagenet` folder is designed for dense datasets.
* Because both training process and predicting process rely on Tensorflow and `tfrecords` format, before running MACH, users need to first convert their datasets to `tfrecords` format specified in `save_to_tfrecords` function in `imagenet/util.py`. This function essentially reads sparse format data line by line, creates an empty Numpy array, fill in values to corresponding indexes, and writes results into `tfrecords` format. The new file may be larger than the original file because the densifing operation. **Feature index and label must starts from 0.**
* After the conversion finished, users will need to modify `NUM_FEATURES`, `NUM_CLASSES`, `TRAIN_FILE`, `TEST_FILE` in `imagenet_demo.py` to accommodate their datasets. If the user wishes to only perform training or predicting, the user can modify `train_imagenet.py` and `predict_imagenet.py` in a similar manner.
* Running MACH will be similar to the tutorials shown in Quickstart section.
