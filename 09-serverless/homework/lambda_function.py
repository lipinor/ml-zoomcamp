import numpy as np

from keras_image_helper import create_preprocessor
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request

from PIL import Image


# Gets interpreter
interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

# Gets the input: the part of the network that takes in the array X
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

# Gets the output: the part of the network with final predictions
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img):
    x = np.array(img) * 1./255
    X = np.array([x])
    X = X.astype('float32')

    return X


def predict(X):
    """Makes a prediction
    """
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)

    return float(pred[0,0])


def lambda_handler(event: dict, context) -> dict:
    """function invoked by the AWS Lambda environment

    Args:
        event (dict): contains all the information we pass to the lambda 
        function in our request
        context (): this parameter will not be used explicitly in this script, 
        but it is necessary. 

    Returns:
        dict: dictionary with predictions labelled for each class.
    """
    url = event['url']

    img_downloaded = download_image(url)
    img_prepared = prepare_image(img_downloaded, (150, 150))

    X = preprocess_image(img_prepared)

    pred = predict(X)

    result = {
        "prediction": pred,
    }

    return result