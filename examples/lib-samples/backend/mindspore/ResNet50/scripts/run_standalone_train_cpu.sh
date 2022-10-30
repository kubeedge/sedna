#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 2 ]
then
    echo "Usage: sh run_standalone_train_cpu.sh [DATASET_PATH] [MODEL_SAVE_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
export BACKEND_TYPE="MINDSPORE"
export DEVICE_CATEGORY="CPU"

if [ ! -d $PATH1 ]
then
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: MODEL_SAVE_PATH=$PATH2 is not a directory"
exit 1
fi

ulimit -u unlimited

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp *.py ./train
cp scripts/*.sh ./train
cp -r src ./train
cd ./train || exit
echo "start training for CPU"
env > env.log
python3 train.py --device_target="CPU" --dataset_path=$PATH1 --model_save_path=$PATH2
cd ..
