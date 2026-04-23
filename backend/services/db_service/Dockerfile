FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update
COPY requirements.txt /workspace/requirements.txt
COPY service.py /workspace/service.py
# COPY settings.json /workspace/settings.json

RUN pip install -r /workspace/requirements.txt

CMD [ "python" ,"service.py"]