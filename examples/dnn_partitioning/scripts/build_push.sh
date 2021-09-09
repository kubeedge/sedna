#!/bin/bash
export PATH=$PATH:/usr/local/go/bin
export GOPATH="$HOME/go"
export GOINSECURE="dmitri.shuralyov.com"
export GOPRIVATE=*
cd /home/ansjin/git_repos/sedna/hack
./update-codegen.sh\

cd /home/ansjin/git_repos/sedna
make gmimage
make lcimage
make crds

docker tag kubeedge/sedna-gm:v0.3.0 registry-cbu.huawei.com/kubeedge/sedna-gm:v0.3.1
docker tag kubeedge/sedna-lc:v0.3.0 registry-cbu.huawei.com/kubeedge/sedna-lc:v0.3.1
docker push registry-cbu.huawei.com/kubeedge/sedna-gm:v0.3.1
docker push registry-cbu.huawei.com/kubeedge/sedna-lc:v0.3.1

cd /home/ansjin/git_repos/sedna/examples/
./build_image.sh

docker tag kubeedge/sedna-example-dnn-partitioning-alex-net-edge:v0.3.0 registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-edge:v0.3.0
docker tag kubeedge/sedna-example-dnn-partitioning-alex-net-cloud:v0.3.0 registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-cloud:v0.3.0
docker push registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-cloud:v0.3.0
docker push registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-edge:v0.3.0


docker pull registry-cbu.huawei.com/kubeedge/sedna-gm:v0.3.1
docker pull registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-cloud:v0.3.0

docker pull registry-cbu.huawei.com/kubeedge/sedna-lc:v0.3.1
docker pull registry-cbu.huawei.com/kubeedge/sedna-example-dnn-partitioning-alex-net-edge:v0.3.0