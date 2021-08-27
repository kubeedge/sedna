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
    echo "Usage: sh run_test_cpu.sh [DATASET_PATH] [CHECKPOINT_PATH]"
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

if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
dirPath=`dirname $PATH2`
ckpt=`basename $PATH2`
export MODEL_URL=$dirPath
export MODEL_NAME=$ckpt

if [ -d "test" ];
then
    rm -rf ./test
fi
mkdir ./test
cp ../*.py ./test
cp *.sh ./test
cp -r ../src ./test
cp -r ../src ./test
cd ./test || exit
env > env.log
echo "start test for CPU"
python test.py --device_target="CPU" --dataset_path=$PATH1 --checkpoint_path=$PATH2 &> log &
cd ..
