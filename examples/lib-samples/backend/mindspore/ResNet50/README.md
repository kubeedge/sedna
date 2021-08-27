# Resnet Example with Mindspore Backend
This document describes how to use the mindspore backend to train Resnet-50 network with the cifar-10 dataset

## Preparatory Stage
### Prepare Dataset
In this example, We need to prepare the cifar10 dataset in advance, and put it into `/home/sedna/examples/backend/mindspore/resnet/`
```bash
cd /home/sedna/examples/backend/mindspore/resnet
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```
### Parameters
you can change the parameters of the model in `src/config.py`

## Modeling Stage
This example support CPU and NPU, you can follow these steps for training, testing and inference
### Train
> * CPU
>```bash
>  bash scripts/run_standalone_train_cpu.sh [DATASET_PATH] [MODEL_SAVE_PATH]
>  # model_save_path must be ABSOLUTE PATH
>  # The log message would be showed in the terminal
>  # The ckpt file would be saved in [MODEL_SAVE_PATH]
>```
> * NPU
>```bash
>  bash scripts/run_standalone_train.sh [DATASET_PATH] [MODEL_SAVE_PATH]
>  # [MODEL_SAVE_PATH] must be ABSOLUTE PATH
>  # The log message would be saved to scripts/train/log
>  # The ckpt file would be saved in [MODEL_SAVE_PATH]
>```
###Test
> * CPU
>```bash
>  bash scripts/run_test_cpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
>  # [CHECKPOINT_PATH] must be ABSOLUTE PATH
>  # The log message would be saved to scripts/test/log
>```
> * NPU
>```bash
>  bash scripts/run_test.sh [DATASET_PATH] [CHECKPOINT_PATH]
>  # [CHECKPOINT_PATH] must be ABSOLUTE PATH
>  # The log message would be saved to scripts/test/log
>```
###Infer
>```bash
>  bash scripts/run_infer.sh [IMAGE_PATH] [CHECKPOINT_PATH]
>  # [CHECKPOINT_PATH] must be ABSOLUTE PATH
>  # The log message would be saved to scripts/infer/log
>```




















