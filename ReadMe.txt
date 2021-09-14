—READ ME FILE—



Read me file for the object detection system——————



Warning: all the process MUST be done in terminal… and please leave the structure of directory



RUNNING THE CONTAINER——



From this directory, build the docker file as follows (this normally takes a while to download the all depencies):



sudo docker build -t objectdet/od-docker .





Run this container:



sudo docker run -p 5000:5000  objectdet/od-docker




-----------


Command to run the TensorFlow serving on gpu: 

—REMEMBER TO OPEN THE TERMINAL FROM THE SAVED_MODEL DIRECTORY AND RUN THIS COMMAND—


sudo docker run --gpus all  -p 8500:8500 --mount type=bind,source=/home/msc1/Desktop/ObjectDetection/saved_model,target=/models/od -e MODEL_NAME=od -t tensorflow/serving:latest-gpu
