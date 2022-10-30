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
    echo "Usage: sh run_infer.sh [IMAGE_PATH] [CHECKPOINT_PATH]"
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


if [ ! -f $PATH1 ]
then
    echo "error: IMAGE_PATH=$PATH1 is not a directory"
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
export BACKEND_TYPE="MINDSPORE"
export DEVICE_CATEGORY="CPU"

if [ -d "infer" ];
then
    rm -rf ./infer
fi
mkdir ./infer
cp *.py ./infer
cp scripts/*.sh ./infer
cp -r src ./infer
cd ./infer || exit
env > env.log
echo "start inference for device $DEVICE_ID"
python3 inference.py --image_path=$PATH1 --checkpoint_path=$PATH2
cd ..
