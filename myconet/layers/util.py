import numpy as np

from ..logger import Logger

"""

This file contains a collection of weight/bias generators
Each activation function may be different, and must be selected correctly
Returns a numpy array of weights, input/output shape are a tuple[int]
The functions are for a range of layer types, e.g. Dense/Convoluted
Always return float32s

"""

class ReLU:
    @staticmethod
    def weights(input_shape, output_shape):
        # He / Kaiming initialization (ReLU-family activations)
        fan_in = np.prod(input_shape)
        std = np.sqrt(2.0 / fan_in)
        return (np.random.randn(*output_shape, *input_shape) * std).astype(np.float32)

    @staticmethod
    def biases(input_shape, output_shape):
        return np.zeros(output_shape, dtype=np.float32)


class Sigmoid:
    @staticmethod
    def weights(input_shape, output_shape):
        # Xavier / Glorot initialization (sigmoid/tanh)
        fan_in = np.prod(input_shape)
        fan_out = np.prod(output_shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(*output_shape, *input_shape)).astype(np.float32)

    @staticmethod
    def biases(input_shape, output_shape):
        return np.zeros(output_shape, dtype=np.float32)


activation_lookup = {
    1: ReLU,
    2: Sigmoid,
}

def weight_init(activation, input_shape, output_shape):
    activation_type = activation_lookup[activation]
    Logger().log(f"Initializing weights for a '{activation_type.__name__}' activated layer")
    return activation_type.weights(input_shape, output_shape)

def bias_init(activation, input_shape, output_shape):
    activation_type = activation_lookup[activation]
    Logger().log(f"  Initializing biases for a '{activation_type.__name__}' activated layer")
    return activation_type.biases(input_shape, output_shape)