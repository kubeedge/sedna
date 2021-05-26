FROM python:3.6

RUN mkdir /code
COPY ./lib/requirements.txt /code
RUN pip install -r /code/requirements.txt

ENV PYTHONPATH "/code/lib"

COPY ./lib /code/lib


ENTRYPOINT ["python", "lib/sedna/service/run_kb.py"]

