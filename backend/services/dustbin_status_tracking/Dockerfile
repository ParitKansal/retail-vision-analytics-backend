FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY service.py /workspace/service.py

RUN pip install -r /workspace/requirements.txt

CMD [ "python" ,"service.py"]
