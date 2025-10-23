FROM python:3.9-slim

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt && \
    pip install transformers==4.46.3 && \
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install accelerate && \
    pip install sentencepiece && \
    pip install numpy==1.24.4

ENV PYTHONPATH="/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]

COPY examples/joint_inference/answer_generation_inference/little_model/little_model.py  /home/work/infer.py
COPY examples/joint_inference/answer_generation_inference/little_model/interface.py  /home/work/interface.py

CMD ["infer.py"]
