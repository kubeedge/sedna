FROM python:3-alpine

RUN mkdir /code
COPY ./lib/requirements.txt /code
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/code/lib"

COPY ./lib /code/lib


ENTRYPOINT ["python", "lib/sedna/service/run_kb.py"]

