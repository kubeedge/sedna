docker build -f /home/lsq/sedna/examples/lifelong-learning-robo-rfnet.Dockerfile --build-arg "HTTP_PROXY=http://10.78.7.206:3128/" \
    --build-arg "HTTPS_PROXY=http://10.78.7.206:3128/" \
    --build-arg "NO_PROXY=localhost,127.0.0.1,.example.com" -t kubeedge/sedna-robo:v0.2.0 ..
