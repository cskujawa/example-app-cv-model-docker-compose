import io
from typing import ByteString, Callable
import numpy as np
import numpy.typing as npt
import streamlit as st
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess_input, decode_predictions as efficientnet_decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input, decode_predictions as resnet_decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input, decode_predictions as inception_decode_predictions
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess_input, decode_predictions as mobilenet_decode_predictions
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess_input, decode_predictions as densenet_decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input, decode_predictions as vgg16_decode_predictions

IMAGENET_INPUT_SIZES = {
    "EfficientNetB0": (224, 224),
    "ResNet50": (224, 224),
    "InceptionV3": (299, 299),
    "MobileNet": (224, 224),
    "DenseNet121": (224, 224),
    "VGG16": (224, 224),
}

def bytes_to_array(image_bytes: ByteString) -> npt.ArrayLike:
    """Converts image stored in bytes into a Numpy array

    Args:
        image_bytes (ByteString): Image stored as bytes

    Returns:
        npt.ArrayLike: Image stored as Numpy array
    """
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

def prepare_image(image_array: npt.ArrayLike, input_size: tuple, _model_preprocess: Callable) -> npt.ArrayLike:
    """Prepares image for prediction by resizing and applying preprocessing

    Args:
        image_array (npt.ArrayLike): Image as a Numpy array
        input_size (tuple): Target input size for the model
        _model_preprocess (Callable): Preprocessing function for the model

    Returns:
        npt.ArrayLike: Preprocessed image
    """
    img = Image.fromarray(image_array.astype('uint8')).resize(input_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = _model_preprocess(img_array)
    return img_array

@st.cache_resource
def load_efficientnet() -> tf.keras.Model:
    model = EfficientNetB0(weights="imagenet", include_top=True)
    return model

@st.cache_resource
def load_resnet() -> tf.keras.Model:
    model = ResNet50(weights="imagenet", include_top=True)
    return model

@st.cache_resource
def load_inception() -> tf.keras.Model:
    model = InceptionV3(weights="imagenet", include_top=True)
    return model

@st.cache_resource
def load_mobilenet() -> tf.keras.Model:
    model = MobileNet(weights="imagenet", include_top=True)
    return model

@st.cache_resource
def load_densenet() -> tf.keras.Model:
    model = DenseNet121(weights="imagenet", include_top=True)
    return model

@st.cache_resource
def load_vgg16() -> tf.keras.Model:
    model = VGG16(weights="imagenet", include_top=True)
    return model

SUPPORTED_MODELS = {
    "EfficientNetB0": {
        "load_model": load_efficientnet,
        "input_size": IMAGENET_INPUT_SIZES["EfficientNetB0"],
        "preprocess_input": efficientnet_preprocess_input,
        "decode_predictions": efficientnet_decode_predictions,
    },
    "ResNet50": {
        "load_model": load_resnet,
        "input_size": IMAGENET_INPUT_SIZES["ResNet50"],
        "preprocess_input": resnet_preprocess_input,
        "decode_predictions": resnet_decode_predictions,
    },
    "InceptionV3": {
        "load_model": load_inception,
        "input_size": IMAGENET_INPUT_SIZES["InceptionV3"],
        "preprocess_input": inception_preprocess_input,
        "decode_predictions": inception_decode_predictions,
    },
    "MobileNet": {
        "load_model": load_mobilenet,
        "input_size": IMAGENET_INPUT_SIZES["MobileNet"],
        "preprocess_input": mobilenet_preprocess_input,
        "decode_predictions": mobilenet_decode_predictions,
    },
    "DenseNet121": {
        "load_model": load_densenet,
        "input_size": IMAGENET_INPUT_SIZES["DenseNet121"],
        "preprocess_input": densenet_preprocess_input,
        "decode_predictions": densenet_decode_predictions,
    },
    "VGG16": {
        "load_model": load_vgg16,
        "input_size": IMAGENET_INPUT_SIZES["VGG16"],
        "preprocess_input": vgg16_preprocess_input,
        "decode_predictions": vgg16_decode_predictions,
    },
}
