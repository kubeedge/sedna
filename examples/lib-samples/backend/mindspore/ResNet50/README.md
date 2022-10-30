# Resnet Example with Mindspore Backend
This document describes how to use the mindspore backend to train Resnet-50 network with the cifar-10 dataset.

## Script Description

### Script and Sample Code
```shell
└──ResNet50
  ├── README.md
  ├── scripts
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_eval_cpu.sh                    # launch cpu evaluation
    ├── run_infer.sh                       # launch cpu inference
    ├── run_standalone_train.sh            # launch ascend standalone training
    ├── run_standalone_train_cpu.sh        # launch cpu training
  ├── src
    ├── config.py                          # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── resnet.py                          # resnet backbone, including resnet50 and resnet101 and se-resnet50
  ├── inference.py                         # Entrance to inference
  ├── interface.py                         # Implements class "Estimator"
  ├── eval.py                              # Entrance to evaluation
  ├── train.py                             # Entrance to training
```

## Script Parameters

Parameters for both training and evaluation can be set in `config.py`.


```bash
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "poly"           # decay mode can be selected in steps, ploy and default
"lr_init": 0.01,                  # initial learning rate
"lr_end": 0.00001,                # final learning rate
"lr_max": 0.1,                    # maximum learning rate
```

## Preparatory Stage
### Prepare Dataset
In this example, we need to prepare the cifar10 dataset in advance, and put it into `/home/sedna/examples/backend/mindspore/resnet/`.
```bash
cd /home/sedna/examples/lib-samples/backend/mindspore/ResNet50
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```
### Parameters
you can change the parameters of the model in `src/config.py`.

## Modeling Stage
This example support CPU and NPU, you can follow these steps for training, testing and inference.

### Training
* #### Running on CPU
```bash
 bash scripts/run_standalone_train_cpu.sh [DATASET_PATH] [MODEL_SAVE_PATH]
 # model_save_path must be ABSOLUTE PATH
 # The log message would be showed in the terminal
 # The ckpt file would be saved in [MODEL_SAVE_PATH]
```
* #### Runing on NPU
```bash
 bash scripts/run_standalone_train.sh [DATASET_PATH] [MODEL_SAVE_PATH]
 # [MODEL_SAVE_PATH] must be ABSOLUTE PATH
 # The log message would be saved to scripts/train/log
 # The ckpt file would be saved in [MODEL_SAVE_PATH]
```

### Evaluation
* #### Running on CPU
```bash 
 bash scripts/run_eval_cpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
 # [CHECKPOINT_PATH] must be ABSOLUTE PATH
 # The log message would be saved to scripts/test/log
```
* #### Running on NPU
```bash
 bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
 # [CHECKPOINT_PATH] must be ABSOLUTE PATH
 # The log message would be saved to scripts/test/log
```

### Inference
```bash
 bash scripts/run_infer.sh [IMAGE_PATH] [CHECKPOINT_PATH]
 # [CHECKPOINT_PATH] must be ABSOLUTE PATH
 # The log message would be saved to scripts/infer/log
```