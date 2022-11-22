import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy

preprocessor = create_preprocessor('xception', target_size=(299, 299))

# Loading the model
interpreter = tflite.Interpreter(model_path='clothing-model-v4.tflite')
interpreter.allocate_tensors()

# Input details
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

# Output details
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def predict(X: numpy.ndarray) -> numpy.ndarray:
    """Make prediction for a single target

    Args:
        X (numpy.ndarray): preprocessed image

    Returns:
        numpy.ndarray: array containing predictions for each class
    """
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0]


# Classes' labels
labels = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]


def decode_predictions(pred: numpy.ndarray) -> dict:
    """Decodes predictions for each class according to label mapping

    Args:
        pred (numpy.ndarray): array with predictions

    Returns:
        dict: dictionary with predictions labelled for each class.
    """
    result = {c: float(p) for c, p in zip(labels, pred)}
    return result


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
    X = preprocessor.from_url(url)
    preds = predict(X)
    results = decode_predictions(preds)
    
    return results
