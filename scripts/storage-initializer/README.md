storage initializer download script

### usage
1. s3 with public access:
```shell
python3 download.py s3://models/classification/model.tar.gz /tmp/models/
# we then download model.tar.gz and extract it into /tmp/models/

```
2. s3 with ak/sk:
```shell
export S3_ENDPOINT_URL=https://play.min.io
export ACCESS_KEY_ID=Q3AM3UQ867SPQQA43P2F
export SECRET_ACCESS_KEY=zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG
python3 download.py s3://datasets/mnist /tmp/mnist
# we then download the content of mnist directory into /tmp/mnist/

```
3. http server:
```shell
python3 download.py http://192.168.3.20/model.pb /tmp/models/
# we then download model.pb into /tmp/models/
```

4. multi downloads:
```shell
python3 download.py http://192.168.3.20/model.pb /tmp/models/ s3://datasets/mnist /tmp/mnist/
# we then download model.pb into /tmp/models/
# and mnist into /tmp/mnist
```

5. indirect download(only support s3-compatible):
the content of `s3://datasets/mnist-index.txt`:

```text
# this first uncomment line is the directory
s3://datasets
mnist/0.jpeg
mnist/1.jpeg
mnist/2.jpeg
```

```shell
export S3_ENDPOINT_URL=https://play.min.io
export ACCESS_KEY_ID=Q3AM3UQ867SPQQA43P2F
export SECRET_ACCESS_KEY=zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG

# note the char @ indicates it's indirect
python3 download.py s3://datasets/mnist-index.txt @/tmp/mnist
# then only download the mnist/{0,1,2}.jpeg into /tmp/mnist/

```
