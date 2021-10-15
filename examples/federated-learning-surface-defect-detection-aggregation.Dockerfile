FROM python:3.7.7 as builder

# install sedna 
COPY ./lib /home/lib
RUN cd /home/lib && python setup.py bdist_wheel && pip install dist/sedna*.whl

# remove unused dependencies of sedna
RUN pip uninstall -y setuptools && pip uninstall -y zipp \
	&& pip uninstall -y importlib-metadata

FROM python:3.7.7-slim

COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages

WORKDIR /home/work
COPY examples/federated_learning/surface_defect_detection/aggregation_worker/  /home/work/

ENTRYPOINT ["python", "aggregate.py"]
