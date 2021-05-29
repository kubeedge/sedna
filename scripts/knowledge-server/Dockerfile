FROM python:3.6

RUN mkdir /code
COPY ./lib/requirements.txt /code
RUN pip install -r /code/requirements.txt
RUN pip install python-multipart
RUN pip install SQLAlchemy
RUN pip install aiofiles
ENV PYTHONPATH "/code/lib"

COPY ./lib /code/lib


ENTRYPOINT ["python", "/code/lib/sedna/service/run_kb.py"]

