cd $(cd $(dirname ${BASH_SOURCE[0]}); pwd)
docker build -f ji-little.Dockerfile -t kubeedge/sedna-example-ji-little:v0.1.0 ..
docker build -f ji-big.Dockerfile -t kubeedge/sedna-example-ji-big:v0.1.0 ..

docker build -f fl-agg.Dockerfile -t kubeedge/sedna-example-fl-agg:v0.1.0 ..
docker build -f fl-train.Dockerfile -t kubeedge/sedna-example-fl-train:v0.1.0 ..

docker build -f il.Dockerfile -t kubeedge/sedna-example-il:v0.1.0 ..
