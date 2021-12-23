FROM python:3.9-buster
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

