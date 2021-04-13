storage initializer download script

### usage
1. s3 with public access:
```shell
python3 download.py s3://models/classification/model.tar.gz /tmp/models/
# we then download model.tar.gz and extract it into /tmp/models/

```
2. s3 with ak/sk:
```shell
export AWS_ENDPOINT_URL=https://play.min.io
export AWS_ACCESS_KEY_ID=Q3AM3UQ867SPQQA43P2F
export AWS_SECRET_ACCESS_KEY=zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG
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
