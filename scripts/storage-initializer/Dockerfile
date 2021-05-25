FROM python:3-alpine

RUN mkdir /code
COPY requirements.txt /code
COPY download.py /code
WORKDIR /code
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "download.py"]
