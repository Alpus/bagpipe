FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Alexander Pushin <work@apushin.com>

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt
