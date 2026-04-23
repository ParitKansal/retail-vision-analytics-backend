FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

CMD ["python", "service.py"]