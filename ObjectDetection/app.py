# importing all the libraries that will need for developing the flask app and 
# for model inference with tensorflow serving and grpc for interaction between the app the tensorflow serving
#numpy for converting in array
from flask import Flask, render_template, request, redirect, jsonify, make_response, flash, url_for
from flask.config import Config
from flask.helpers import send_from_directory
import numpy as np
import os
from werkzeug.utils import secure_filename
import sys
import tensorflow as tf
from PIL import Image
import time
import cv2
from absl import app, logging
import grpc 
from grpc.beta import implementations

#importing from tensorflow the serving api that will be need for for prediction service
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
#this is for picking the label files and files from core folder in the directory
from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

sys.path.append("..")
#giving the path to retrieve the model will be used, and also defining the numbers 90 in this model, which is 90
PATH_TO_LABEL = "./data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90 

label_map = label_map_util.load_labelmap(PATH_TO_LABEL)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, 
                                                            use_display_name=True)

category_index = label_map_util.create_category_index(categories)

#print the num of available in the machine(this will also specificy that gpu must primary and in case of error to use CPU)
# and how long it takes to load the model
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

print('Loading model...', end='')
start_time = time.time()

end_time = time.time()
elapsed_time = end_time - start_time
print('Done| Took {} seconds'.format(elapsed_time))

#defining the flask app and i also include a secret key which will keep the client-side sessions secure.
app = Flask(__name__)
app.config['SECRET_KEY'] = "abc123"

#config the app to know where to instore the image when someone uploads an image, and to accept or reduce only few format of image 
app.config["IMAGE_UPLOADS"] = "./static/uploads/img/upload"
app.config["ALLOWED_IMAGE_EXTENSIONS"] =["PNG", "JPG", "JPEG"]

# this is about defining which file(in term of the name and format) that the application will accept to upload
def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

# defining the host and the port which will be the interface that will be communicating with the tensorflow serving, 
# and this will be done by the function get_stub
def get_stub(host='127.0.0.1', port='8500'):
    channel = grpc.insecure_channel('127.0.0.1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub
# this part is about the image processing, in fact the image numpy array will be converted into tensor
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    
    return tensor 
# this part is about the inferences, first i set up the channel where the gprc will listen to for 
# communication with tensorflow serving, i defined the name of the nodel 
def inference(frame, stub, model_name='od'):
    channel = grpc.insecure_channel('localhost:8500', options=(('grpc.enable_http_proxy', 0),))
    print("Channel: ", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print("Stub: ", stub)
    request = predict_pb2.PredictRequest()
    print("Request: ", request)
    request.model_spec.name = 'od'

# this steps is about passing the image 
# to the visualize_boxes_and_labels_on_image_array function which places the boxes and scores on the image
#this will also return the name of the object classified and the percentage.

    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0)

    image_np = load_image_into_numpy_array(image)

    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)
    treshold= 0.60
    frame = viz_utils.visualize_boxes_and_labels_on_image_array(image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh= treshold,
                    agnostic_mode=False)
    object_det = []
    for idx in output_dict['detection_scores']:
   	    if idx > treshold:
               print(idx)
               print(np.where(output_dict['detection_scores'] == idx)[0][0])
               print(output_dict['detection_classes'][np.where(output_dict['detection_scores'] == idx)[0][0]])
               det = output_dict['detection_classes'][np.where(output_dict['detection_scores'] == idx)[0][0]]
               print(category_index[det]['name'])
               name_det = category_index[det]['name']
               object_det.append((category_index[det]['name'], idx))
    print(object_det)    
    return frame, object_det


    
#This is the first route of the flask app, which contains the home page(index.html)
#so when we start this app, the appp will the the index template from the template directory

@app.route("/")
def index():

    return render_template("index.html")

#This app route is about uploading the image, so with the methods post, 
# this function will request the file submitted for validation, in fact filenamw will be checked, 
# extensions will checked also, in case one of this condition not meet the requirements, the app will return to 
# the index page. In case all conditions are meet, the image will be save(already define in the app.config).

@app.route("/upload", methods=["POST"])   
def upload():

    if request.files:

        image = request.files["image"]

        if image.filename == "":
            print("Image must have a filename")
            return render_template("index.html")

        if not allowed_image(image.filename):
            flash("That image extensions is not allowed")
            return render_template("index.html")

        else:
            filename = secure_filename(image.filename)

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

        print("Image Saved")

            
    return redirect(url_for("upload_file", filename=filename))


#This is the last route of the app, it about displaying the original image submitted and the inferenced image.
# this image will be display on the results template
@app.route("/results/<filename>")
def upload_file(filename):
    PATH_TO_TEST_IMAGE_DIR = app.config["IMAGE_UPLOADS"]
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGE_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12,8)

    stub = get_stub()

    for image_path in TEST_IMAGE_PATHS:
        image_np = np.array(Image.open(image_path))
        image_np_inferenced, object_det = inference(image_np, stub)
        im = Image.fromarray(image_np_inferenced)
        im.save('static/uploads/img/detection/' + filename)
        
    return render_template("results.html", filename=filename, result = object_det)

#Extra app for the info page, this page will give information about the model used and extra information 
@app.route("/info")
def information():

    return render_template("info.html")

# end the flask app, i will define the host ip and the port 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
