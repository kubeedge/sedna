FROM python:3.9-slim

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt && \
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
    transformers==4.46.3 \
    accelerate \
    sentencepiece \
    numpy==1.24.4 \
    openai

ENV PYTHONPATH="/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]

COPY examples/joint_inference/answer_generation_inference/big_model/big_model.py  /home/work/infer.py
COPY examples/joint_inference/answer_generation_inference/big_model/interface.py  /home/work/interface.py

CMD ["infer.py"]
