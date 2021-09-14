FROM ubuntu
FROM python:3.7.6
FROM tensorflow/tensorflow:latest
ADD ObjectDetection  /ObjectDetection

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install Flask
RUN pip install tensorflow
RUN pip install Flask-WTF
RUN pip install numpy==1.19.2
RUN pip install opencv-python
RUN pip install pafy
RUN pip install youtube_dl
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install tensorflow-serving-api
RUN pip install grpcio
RUN pip install grpcio-tools

WORKDIR "./ObjectDetection"
CMD [ "python", "./app.py" ]


