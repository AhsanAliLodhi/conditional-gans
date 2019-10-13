FROM python:3
ADD . /app
WORKDIR /app
RUN apt install gcc
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
CMD python server.py
