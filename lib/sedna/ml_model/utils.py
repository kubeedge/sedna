import tensorflow as tf

from sedna.common.config import BaseConfig


def load_model():
    model_url = BaseConfig.model_url
    return tf.keras.models.load_model(model_url)


def save_model(model):
    model_url = BaseConfig.model_url
    model.save(model_url)
